#!/usr/bin/env python3
"""
Evaluate a trained TELGEN checkpoint against the TRUE LP optimum (Gurobi), using the
metric definitions from the paper's ER table:

    x_raw   = relu(model(...)[:, -1])                    raw predicted solution (may be infeasible)
    x_proj  = relu(model(..., enforce_feasibility=True)) feasibility-projected solution
    f_opt   = Gurobi optimum of   min c_norm^T x   s.t.  A x <= b,  0 <= x <= 1

    OGap    = | f0(x_proj) - f_opt | / |f_opt|                     objective gap of the projected solution
    CGap    = sum_row relu(A x_raw - b)                            total constraint violation of the raw solution
    OnoCGap = | f0(x_raw)/alpha - f_opt | / |f_opt|,               raw solution scaled down to feasibility,
              alpha = max_row (A x_raw / b),  clamped to >= 1      then objective gap  (paper: "scaled by the
                                                                  maximum ratio of any constraint violation")

  - c is normalised exactly as in data/dataset.py:  c_norm = c / (max|c| + 1e-10)
  - constraints are pre-normalised so b == 1 for every row.

Requires gurobipy. Operates directly on the raw instance files
(instance_ER_{split}_p{p}_n{n}.pkl.gz), so no processed cache is needed.

Example:
  python eval_er_gurobi_metric.py \
    --checkpoint checkpoints/<run>/best_model.pth \
    --raw_dir ./data/raw \
    --p_values 0.1 0.3 0.5 0.7 0.9 --n_values 200 500 1000 2000 \
    --max_instances 50 --output_csv gurobi_metric.csv
"""
import argparse
import gzip
import os
import pickle
import time

import numpy as np
import torch
from torch_geometric.data import HeteroData

from models.hetero_gnn import TripartiteHeteroGNN_
from data.data_preprocess import HeteroAddLaplacianEigenvectorPE

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover
    raise SystemExit("gurobipy is required for this script: %s" % exc)


# --- paper Table (ER, test) for side-by-side reference -------------------------
PAPER_TABLE = {
    (200, 0.1): (3.88, 6.14, 0.30), (200, 0.5): (3.31, 5.17, 0.48), (200, 0.9): (3.18, 5.48, 0.58),
    (500, 0.1): (3.71, 5.25, 0.41), (500, 0.5): (2.90, 5.23, 0.44), (500, 0.9): (3.48, 5.53, 0.49),
    (1000, 0.1): (3.53, 5.00, 0.64), (1000, 0.5): (3.34, 5.21, 0.48), (1000, 0.9): (3.19, 5.52, 0.43),
    (2000, 0.1): (3.82, 5.02, 0.71), (2000, 0.5): (3.31, 5.17, 0.48), (2000, 0.9): (3.18, 5.48, 0.58),
}


def build_hetero(A, b, c):
    """Reproduce the HeteroData construction from data/dataset.py (without solving the LP)."""
    A = A.to(torch.float32)
    b = b.to(torch.float32)
    c = c.to(torch.float32)
    c = c / (c.abs().max() + 1e-10)
    row, col = torch.where(A != 0)
    val = A[row, col]
    return HeteroData(
        cons={'x': torch.cat([A.mean(1, keepdim=True), A.std(1, keepdim=True)], dim=1)},
        vals={'x': torch.cat([A.mean(0, keepdim=True), A.std(0, keepdim=True)], dim=0).T},
        obj={'x': torch.cat([c.mean(0, keepdim=True), c.std(0, keepdim=True)], dim=0)[None]},
        cons__to__vals={'edge_index': torch.vstack(torch.where(A != 0)), 'edge_attr': A[torch.where(A != 0)][:, None]},
        vals__to__cons={'edge_index': torch.vstack(torch.where(A.T != 0)), 'edge_attr': A.T[torch.where(A.T != 0)][:, None]},
        vals__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[1]), torch.zeros(A.shape[1], dtype=torch.long)]), 'edge_attr': c[:, None]},
        obj__to__vals={'edge_index': torch.vstack([torch.zeros(A.shape[1], dtype=torch.long), torch.arange(A.shape[1])]), 'edge_attr': c[:, None]},
        cons__to__obj={'edge_index': torch.vstack([torch.arange(A.shape[0]), torch.zeros(A.shape[0], dtype=torch.long)]), 'edge_attr': b[:, None]},
        obj__to__cons={'edge_index': torch.vstack([torch.zeros(A.shape[0], dtype=torch.long), torch.arange(A.shape[0])]), 'edge_attr': b[:, None]},
        gt_primals=torch.zeros(A.shape[1], 16),
        obj_value=torch.tensor(0.0),
        obj_const=c,
        A_row=row, A_col=col, A_val=val,
        A_num_row=A.shape[0], A_num_col=A.shape[1], A_nnz=len(val),
        A_tilde_mask=torch.ones(row.shape, dtype=torch.bool), rhs=b)


def project_local(x, A, b, iters=20):
    """Iterative *local* feasibility scaling: each variable is scaled by the tightest
    constraint it participates in (not one global factor). Preserves objective far
    better than global x/alpha_max when only a few links are congested."""
    x = torch.clamp(x, min=0.0).clone()
    mask = (A != 0)
    for _ in range(iters):
        Ax = A @ x
        ratio = torch.where(Ax > b, b / (Ax + 1e-12), torch.ones_like(b))  # [m], <=1 on violated rows
        if (ratio >= 1.0 - 1e-9).all():
            break
        # per-variable multiplier = min ratio over the constraints that variable touches
        m_per_var = torch.ones_like(x)
        rmat = torch.where(mask, ratio[:, None], torch.ones_like(A))  # [m,n]
        m_per_var = rmat.min(dim=0).values
        x = x * m_per_var
    return x


def project_lp(x_raw, A, b, c, time_limit=10.0):
    """Feasibility restoration by LP:  min c^T x  s.t.  A x <= b, 0 <= x <= x_raw.
    Only allows *reducing* the model's flows; keeps the objective as high as possible."""
    A = A.cpu().numpy().astype(np.float64)
    b = b.cpu().numpy().astype(np.float64)
    c = c.cpu().numpy().astype(np.float64)
    ub = np.clip(x_raw.cpu().numpy().astype(np.float64), 0.0, None)
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit)
    x = m.addMVar(shape=A.shape[1], lb=0.0, ub=ub)
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.addMConstr(A, x, '<', b)
    m.optimize()
    if m.SolCount == 0:
        m.dispose()
        return None
    obj = float(m.ObjVal)
    m.dispose()
    return obj


def gurobi_optimum(A, b, c, time_limit=30.0):
    A = A.cpu().numpy().astype(np.float64)
    b = b.cpu().numpy().astype(np.float64)
    c = c.cpu().numpy().astype(np.float64)
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    m.setParam('TimeLimit', time_limit)
    x = m.addMVar(shape=A.shape[1], lb=0.0, ub=1.0)
    m.setObjective(c @ x, GRB.MINIMIZE)
    m.addMConstr(A, x, '<', b)
    m.optimize()
    if m.SolCount == 0:
        m.dispose()
        raise RuntimeError('Gurobi returned no solution (status=%s)' % m.Status)
    obj = float(m.ObjVal)
    m.dispose()
    return obj


def build_model(params, device):
    model = TripartiteHeteroGNN_(
        ipm_steps=params.get('ipm_steps', 16), conv=params.get('conv', 'gcnconv'), in_shape=2,
        pe_dim=params.get('lappe', 0), hid_dim=params.get('hidden', 128),
        num_conv_layers=params.get('num_conv_layers', 2), num_pred_layers=params.get('num_pred_layers', 4),
        num_mlp_layers=params.get('num_mlp_layers', 4), dropout=params.get('dropout', 0.0),
        share_lin_weight=params.get('share_lin_weight', True), share_conv_weight=params.get('share_conv_weight', True),
        use_norm=params.get('use_norm', True), use_res=params.get('use_res', True),
        conv_sequence=params.get('conv_sequence', 'cov')).to(device)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--raw_dir', default='./data/raw')
    ap.add_argument('--split', default='test', choices=['test', 'train'])
    ap.add_argument('--p_values', type=float, nargs='*', default=[0.1, 0.3, 0.5, 0.7, 0.9])
    ap.add_argument('--n_values', type=int, nargs='*', default=[200, 500, 1000, 2000])
    ap.add_argument('--max_instances', type=int, default=50, help='instances per (p,n) file; 0 = all')
    ap.add_argument('--sample_seed', type=int, default=42)
    ap.add_argument('--gurobi_time_limit', type=float, default=30.0)
    ap.add_argument('--enforce_feasibility', dest='enforce_feasibility', action='store_true', default=True)
    ap.add_argument('--no_projection', dest='enforce_feasibility', action='store_false')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--output_csv', default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    params = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    model = build_model(params, device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print("checkpoint: hidden=%s lappe=%s epochs=%s bs=%s best_val_loss=%s | projection=%s" % (
        params.get('hidden'), params.get('lappe'), params.get('epochs'), params.get('batchsize'),
        ckpt.get('best_val_loss') if isinstance(ckpt, dict) else None, args.enforce_feasibility))

    S = int(params.get('ipm_steps', 16))
    lappe = int(params.get('lappe', 0) or 0)
    pe_transform = HeteroAddLaplacianEigenvectorPE(k=lappe) if lappe > 0 else None
    rows = []
    print("\n(all values are %, paper column key: O=OGap C=CGap Oc=OnoCGap)\n")
    for n in args.n_values:
        for p in args.p_values:
            fpath = os.path.join(args.raw_dir, "instance_ER_%s_p%s_n%s.pkl.gz" % (args.split, p, n))
            if not os.path.exists(fpath):
                continue
            with gzip.open(fpath, 'rb') as fh:
                raw = pickle.load(fh)
            rng = np.random.RandomState(args.sample_seed + n + int(round(p * 1000)))
            idx = rng.permutation(len(raw))
            if args.max_instances > 0:
                idx = idx[:args.max_instances]
            og, cg, onoc, tms = [], [], [], []
            for i in idx:
                t = raw[i]
                if not isinstance(t, (tuple, list)) or len(t) < 3:
                    continue
                A0, b0, c0 = t[0], t[1], t[2]
                data = build_hetero(A0, b0, c0)
                if pe_transform is not None:
                    data = pe_transform(data)
                data = data.to(device)
                A = torch.zeros(int(data.A_num_row), int(data.A_num_col), device=device)
                A[data.A_row, data.A_col] = data.A_val
                b = data.rhs
                c = data.obj_const
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    vals_raw, _ = model(data, enforce_feasibility=False)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                tms.append((time.time() - t0) * 1000.0)  # pure model forward, matches paper's timing

                x_raw = torch.relu(vals_raw[:, -1])
                obj_opt = gurobi_optimum(A, b, c, args.gurobi_time_limit)
                denom = max(abs(obj_opt), 1e-12)
                obj_raw = float((c * x_raw).sum())
                Ax_raw = A @ x_raw

                # OGap candidates
                ogap_raw = abs((obj_raw - obj_opt) / denom)                       # raw output
                with torch.no_grad():
                    vals_p, _ = model(data, enforce_feasibility=True)
                x_proj = torch.relu(vals_p[:, -1])
                ogap_proj = abs((float((c * x_proj).sum()) - obj_opt) / denom)     # feasibility-projected output

                # CGap candidates: constraint violation of the RAW output  (b == 1 after normalisation)
                cgap_sum = float(torch.relu(Ax_raw - b).sum())
                cgap_max = float(torch.relu(Ax_raw - b).max())

                # OnoCGap candidates: 3 feasibility-restoration strategies on the RAW output
                alpha = float((Ax_raw / b).max())
                alpha = alpha if alpha > 1.0 else 1.0
                ono_global = abs((obj_raw / alpha - obj_opt) / denom)              # x_raw / alpha_max
                x_loc = project_local(x_raw, A, b)
                ono_local = abs((float((c * x_loc).sum()) - obj_opt) / denom)      # per-variable local scaling
                o_lp = project_lp(x_raw, A, b, c)
                ono_lp = abs((o_lp - obj_opt) / denom) if o_lp is not None else float('nan')  # LP: min c'x s.t. Ax<=b, x<=x_raw

                og.append((ogap_raw, ogap_proj))
                cg.append((cgap_sum, cgap_max))
                onoc.append((ono_global, ono_local, ono_lp))
            if not og:
                continue
            og = np.array(og); cg = np.array(cg); on = np.array(onoc)
            ogap_raw, ogap_proj = og[:, 0].mean() * 100, og[:, 1].mean() * 100
            cgap_sum, cgap_max = cg[:, 0].mean() * 100, cg[:, 1].mean() * 100
            ono_g, ono_l, ono_lp = np.nanmean(on[:, 0]) * 100, np.nanmean(on[:, 1]) * 100, np.nanmean(on[:, 2]) * 100
            pt = PAPER_TABLE.get((n, p))
            ptxt = ("[paper O%.2f C%.2f Oc%.2f]" % pt) if pt else ""
            print("%-13s OGap_raw %5.2f  CGap[sum %5.2f max %5.2f]  restore-OGap[global %5.2f  local %5.2f  lp %5.2f]  %s" % (
                "ER %d,%s" % (n, p), ogap_raw, cgap_sum, cgap_max, ono_g, ono_l, ono_lp, ptxt))
            rows.append((n, p, ogap_raw, ogap_proj, cgap_sum, cgap_max, ono_g, ono_l, ono_lp, np.mean(tms), len(og)))

    if args.output_csv and rows:
        import csv
        with open(args.output_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['n', 'p', 'OGap_raw_pct', 'OGap_proj_pct', 'CGap_sum_pct', 'CGap_max_pct',
                        'restore_global_pct', 'restore_local_pct', 'restore_lp_pct', 'infer_ms', 'n_eval'])
            w.writerows(rows)
        print("\nSaved CSV to %s" % args.output_csv)


if __name__ == '__main__':
    main()
