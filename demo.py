#!/usr/bin/env python3
"""
demo.py — run the trained TELGEN model on the large ER test graphs (n = 1000, 2000)
and report the optimality / feasibility gaps against the LP optimum.

These are the "generalization" test sets: the model was trained only on graphs with
20-100 nodes, and here it is applied to graphs 10-20x larger, unseen during training.

Usage
-----
    python demo.py                          # p=0.5, n=1000, 50 instances (default)
    python demo.py --p 0.9 --n 2000         # a different test set
    python demo.py --n 1000 --p 0.3 --num_instances 100
    python demo.py --list                   # show all available (p, n) test sets

What it prints
--------------
    OGap    objective (optimality) gap of the raw model output vs the LP optimum
    CGap    total constraint violation of the raw model output   (paper's gamma_con)
    OnoCGap objective gap after a cheap per-link feasibility restoration
    time    pure model forward time per instance (ms)

The LP optimum is obtained with Gurobi if `gurobipy` is installed, otherwise with the
bundled interior-point solver in `solver/` (both give the same value on these LPs).

Note: the model works well on n >= 1000 for every density. On the smallest + sparsest
sets (n = 200 or 500 with p = 0.1) many instances are bottleneck-dominated and the
spatial flow distribution degrades — those are not covered by this demo.
"""
import argparse
import glob
import gzip
import os
import pickle
import random
import time

import numpy as np
import torch


def set_seed(seed: int = 0):
    """Pin every RNG so the reported numbers are byte-reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from models.hetero_gnn import TripartiteHeteroGNN_
from eval_er_gurobi_metric import build_hetero, project_local

try:
    from data.data_preprocess import HeteroAddLaplacianEigenvectorPE
except Exception:
    HeteroAddLaplacianEigenvectorPE = None

# paper Table V (ER test), for side-by-side reference
PAPER = {
    (1000, 0.1): (3.53, 5.00, 0.64), (1000, 0.5): (3.34, 5.21, 0.48), (1000, 0.9): (3.19, 5.52, 0.43),
    (2000, 0.1): (3.82, 5.02, 0.71), (2000, 0.5): (3.31, 5.17, 0.48), (2000, 0.9): (3.18, 5.48, 0.58),
}


def lp_optimum(A, b, c):
    """LP optimum of  min c^T x  s.t.  A x <= b, 0 <= x <= 1."""
    A = A.cpu().numpy().astype(np.float64)
    b = b.cpu().numpy().astype(np.float64)
    c = c.cpu().numpy().astype(np.float64)
    try:
        import gurobipy as gp
        from gurobipy import GRB
        m = gp.Model(); m.setParam('OutputFlag', 0); m.setParam('TimeLimit', 30.0)
        m.setParam('Threads', 1); m.setParam('Seed', 0)  # deterministic
        x = m.addMVar(shape=A.shape[1], lb=0.0, ub=1.0)
        m.setObjective(c @ x, GRB.MINIMIZE)
        m.addMConstr(A, x, '<', b)
        m.optimize()
        obj = float(m.ObjVal); m.dispose()
        return obj
    except Exception:
        from solver.linprog import linprog
        sol = linprog(c, A_ub=A, b_ub=b, bounds=(0, 1), method='interior-point')
        return float(sol.fun)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ck.get('args', {})
    model = TripartiteHeteroGNN_(
        ipm_steps=a.get('ipm_steps', 16), conv=a.get('conv', 'gcnconv'), in_shape=2,
        pe_dim=a.get('lappe', 0), hid_dim=a.get('hidden', 64),
        num_conv_layers=a.get('num_conv_layers', 2), num_pred_layers=a.get('num_pred_layers', 4),
        num_mlp_layers=a.get('num_mlp_layers', 4), dropout=a.get('dropout', 0.0),
        share_lin_weight=a.get('share_lin_weight', True), share_conv_weight=a.get('share_conv_weight', True),
        use_norm=a.get('use_norm', True), use_res=a.get('use_res', True),
        conv_sequence=a.get('conv_sequence', 'cov')).to(device)
    model.load_state_dict(ck.get('state_dict', ck), strict=False)
    model.eval()
    lappe = int(a.get('lappe', 0) or 0)
    pe = HeteroAddLaplacianEigenvectorPE(k=lappe) if (lappe > 0 and HeteroAddLaplacianEigenvectorPE) else None
    return model, pe, a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    ap.add_argument('--raw_dir', default='data/raw')
    ap.add_argument('--n', type=int, default=1000, choices=[1000, 2000], help='test graph size')
    ap.add_argument('--p', type=float, default=0.5, help='edge probability (0.1 .. 0.9)')
    ap.add_argument('--num_instances', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0, help='RNG seed; fixed by default for reproducibility')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--list', action='store_true', help='list available test sets and exit')
    args = ap.parse_args()

    set_seed(args.seed)

    if args.list:
        for f in sorted(glob.glob(os.path.join(args.raw_dir, 'instance_ER_test_*.pkl.gz'))):
            print(' ', os.path.basename(f))
        return

    fpath = os.path.join(args.raw_dir, f'instance_ER_test_p{args.p}_n{args.n}.pkl.gz')
    if not os.path.exists(fpath):
        raise SystemExit(f'not found: {fpath}\n(run  python demo.py --list  to see what is available)')

    device = torch.device(args.device)
    model, pe, a = load_model(args.checkpoint, device)
    print(f'model: hidden={a.get("hidden")} lappe={a.get("lappe")} losstype={a.get("losstype")} '
          f'| device={device}')
    print(f'test set: ER n={args.n}, p={args.p}  ({args.num_instances} instances)\n')

    raw = pickle.load(gzip.open(fpath, 'rb'))
    n_eval = min(args.num_instances, len(raw))
    og, cg, onoc, tms = [], [], [], []
    np.random.seed(args.seed)  # re-pin: the Laplacian-PE eigensolver draws its start vector from numpy's RNG
    for t in raw[:n_eval]:
        A0, b0, c0 = t[0], t[1], t[2]
        data = build_hetero(A0, b0, c0)
        if pe is not None:
            data = pe(data)
        data = data.to(device)
        A = torch.zeros(int(data.A_num_row), int(data.A_num_col), device=device)
        A[data.A_row, data.A_col] = data.A_val
        b, c = data.rhs, data.obj_const

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            vals, _ = model(data)                     # [num_paths, ipm_steps]
        if device.type == 'cuda':
            torch.cuda.synchronize()
        tms.append((time.time() - t0) * 1000.0)

        x = torch.relu(vals[:, -1])                   # predicted traffic-split ratios R(p)
        obj_opt = lp_optimum(A, b, c)
        den = max(abs(obj_opt), 1e-12)

        og.append(abs((float((c * x).sum()) - obj_opt) / den))
        cg.append(float(torch.relu(A @ x - b).sum()))
        x_feas = project_local(x, A, b)               # cheap per-link feasibility restoration
        onoc.append(abs((float((c * x_feas).sum()) - obj_opt) / den))

    o, k, oc = np.mean(og) * 100, np.mean(cg) * 100, np.mean(onoc) * 100
    print(f'  {"":8s}{"OGap":>9}{"CGap":>9}{"OnoCGap":>10}{"time/inst":>12}')
    print(f'  {"TELGEN":8s}{o:>8.2f}%{k:>8.2f}%{oc:>9.2f}%{np.mean(tms):>10.2f}ms')
    pt = PAPER.get((args.n, args.p))
    if pt:
        print(f'  {"paper":8s}{pt[0]:>8.2f}%{pt[1]:>8.2f}%{pt[2]:>9.2f}%')
    print(f'\n  ({n_eval} instances, LP optimum via '
          f'{"Gurobi" if _has_gurobi() else "bundled interior-point solver"})')


def _has_gurobi():
    try:
        import gurobipy  # noqa
        return True
    except Exception:
        return False


if __name__ == '__main__':
    main()
