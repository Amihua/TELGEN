#!/usr/bin/env python3
"""Train the ER model with the FULL weighted loss (Trainer.train_: primal + objgap + constraint),
matching the paper's Optuna pipeline -- NOT the release's primal-only SimpleTrainer.
Validation metric = mean objective gap (as in optuna_hyperparameter_tuning_working.py).
"""
import argparse, os, sys, copy, datetime, random
import numpy as np, torch
from torch import optim
from torch.utils.data import DataLoader
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.utils import collate_fn_ip
from models.hetero_gnn import TripartiteHeteroGNN_
from trainer import Trainer
from run_er_single_experiment import create_datasets


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s); np.random.seed(s); random.seed(s)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./data')
    ap.add_argument('--val_seed', type=int, default=2025); ap.add_argument('--val_size', type=int, default=300)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--test_p', type=float, default=0.9); ap.add_argument('--test_n', type=int, default=2000)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--num_conv_layers', type=int, default=2); ap.add_argument('--num_pred_layers', type=int, default=4)
    ap.add_argument('--num_mlp_layers', type=int, default=4); ap.add_argument('--dropout', type=float, default=0.01936460851830677)
    ap.add_argument('--conv', default='gcnconv'); ap.add_argument('--conv_sequence', default='cov')
    ap.add_argument('--epochs', type=int, default=150); ap.add_argument('--batchsize', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.0004590632782570314); ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--ipm_steps', type=int, default=16); ap.add_argument('--ipm_alpha', type=float, default=0.7)
    ap.add_argument('--loss_weight_x', type=float, default=1.0)
    ap.add_argument('--loss_weight_obj', type=float, default=3.43); ap.add_argument('--loss_weight_cons', type=float, default=5.8)
    ap.add_argument('--losstype', default='l2'); ap.add_argument('--lappe', type=int, default=8)
    ap.add_argument('--micro_batch', type=int, default=1)
    ap.add_argument('--num_workers', type=int, default=1)
    ap.add_argument('--share_lin_weight', action='store_true', default=True)
    ap.add_argument('--share_conv_weight', action='store_true', default=True)
    ap.add_argument('--use_norm', action='store_true', default=True)
    ap.add_argument('--use_res', action='store_true', default=True)
    ap.add_argument('--ckpt_dir', default='./checkpoints_weighted')
    args = ap.parse_args()

    set_seed(args.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, val_ds, test_ds = create_datasets(args)
    tl = DataLoader(train_ds, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_ip)
    vl = DataLoader(val_ds, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_ip)
    testl = DataLoader(test_ds, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_ip)

    model = TripartiteHeteroGNN_(
        ipm_steps=args.ipm_steps, conv=args.conv, in_shape=2, pe_dim=args.lappe, hid_dim=args.hidden,
        num_conv_layers=args.num_conv_layers, num_pred_layers=args.num_pred_layers, num_mlp_layers=args.num_mlp_layers,
        dropout=args.dropout, share_lin_weight=args.share_lin_weight, share_conv_weight=args.share_conv_weight,
        use_norm=args.use_norm, use_res=args.use_res, conv_sequence=args.conv_sequence).to(dev)

    def init_w(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight); torch.nn.init.zeros_(m.bias)
    model.apply(init_w)

    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trainer = Trainer(device=dev, loss_target='primal+objgap+constraint', loss_type=args.losstype,
                      micro_batch=args.micro_batch, ipm_steps=args.ipm_steps, ipm_alpha=args.ipm_alpha,
                      loss_weight={'primal': args.loss_weight_x, 'objgap': args.loss_weight_obj, 'constraint': args.loss_weight_cons})

    run = f"er_weighted_h{args.hidden}_lappe{args.lappe}_bs{args.batchsize}_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    ckdir = os.path.join(args.ckpt_dir, run); os.makedirs(ckdir, exist_ok=True)
    best_val = float('inf'); best_state = None
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} | weighted loss 1/{args.loss_weight_obj}/{args.loss_weight_cons}, micro_batch={args.micro_batch}", flush=True)

    for ep in range(1, args.epochs + 1):
        tr = trainer.train_(tl, model, opt)
        avg_tl, p_l, o_l, c_l = tr
        vg, vc, vnc = trainer.eval_metrics_(vl, model)
        v_objgap = float(np.mean(np.abs(np.concatenate([g if isinstance(g, np.ndarray) else np.array([g]) for g in [vg]]))))
        v_consviol = float(np.mean(np.abs(vc)))
        v_adj = float(np.mean(np.abs(vnc)))
        msg = (f"Epoch {ep:03d} | TrainTotal {avg_tl:.6f} (P {p_l:.6f} O {o_l:.6f} C {c_l:.6f}) | "
               f"Val ObjGap {v_objgap:.6f} ConsViol {v_consviol:.6f} AdjObjGap {v_adj:.6f}")
        if v_objgap < best_val:
            best_val = v_objgap
            best_state = copy.deepcopy(model.state_dict())
            torch.save({'state_dict': best_state, 'epoch': ep, 'best_val_loss': best_val, 'args': vars(args)},
                       os.path.join(ckdir, 'best_model.pth'))
            msg += "  <- new best"
        print(msg, flush=True)

    model.load_state_dict(best_state)
    tg, tc, tnc = trainer.eval_metrics_(testl, model)
    print(f"\n=== TEST (p={args.test_p} n={args.test_n}): ObjGap {np.mean(np.abs(tg)):.6f} "
          f"ConsViol {np.mean(np.abs(tc)):.6f} AdjObjGap {np.mean(np.abs(tnc)):.6f} ===", flush=True)
    print(f"ckpt: {os.path.join(ckdir, 'best_model.pth')}  best_val_objgap={best_val:.6f}", flush=True)


if __name__ == '__main__':
    main()
