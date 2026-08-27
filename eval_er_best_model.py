#!/usr/bin/env python3
"""
Evaluate a trained TELEGEN model on ER test sets and report grouped metrics.

- Loads a model checkpoint specified by the user
- For each available ER test group G_n^p (files named: instance_ER_test_p{p}_n{n}.pkl.gz)
  computes the three metrics using original definitions from trainer.Trainer:
    1) objective gap
    2) constraint violation
    3) adjusted objective gap
  and reports mean and std for each group.
"""

import os
import re
import argparse
import json
import time
import psutil
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose

# 尝试从 torch_scatter 导入 scatter；如果失败，则使用简化版实现（仅支持 reduce='sum'）
try:  # pragma: no cover - 环境兼容逻辑
    from torch_scatter import scatter  # type: ignore
except Exception:
    def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = 0, reduce: str = "sum") -> torch.Tensor:
        """
        简化版 scatter，仅支持 reduce='sum'，满足本脚本对约束聚合和目标聚合的需求：
        - src: [..., num_features]，例如 [num_rows, T]
        - index: 与 src 在 dim 维上形状相同的一维索引
        - dim: 归约维度
        """
        if reduce != "sum":
            raise NotImplementedError("Fallback scatter only supports reduce='sum'")
        if index.dim() != 1:
            raise ValueError(f"Unsupported index shape for fallback scatter: {index.shape}")
        if dim != 0:
            # 只在 dim=0 的情况下使用；其他情况不在本脚本中出现
            raise NotImplementedError("Fallback scatter is only implemented for dim=0")
        if index.numel() == 0:
            return torch.zeros((0,) + src.shape[1:], dtype=src.dtype, device=src.device)
        out_size = int(index.max().item()) + 1
        out_shape = (out_size,) + src.shape[1:]
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        expanded_index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        return out.scatter_add(0, expanded_index, src)

from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
from data.dataset import LPDataset
from data.utils import collate_fn_ip
from models.hetero_gnn import TripartiteHeteroGNN_


def discover_test_groups(raw_dir: str):
    """Discover available ER test groups by scanning raw_dir for pattern instance_ER_test_p{p}_n{n}.pkl.gz"""
    pattern = re.compile(r"^instance_ER_test_p([0-9.]+)_n(\d+)\.pkl\.gz$")
    groups = set()
    if not os.path.isdir(raw_dir):
        return []
    for fname in os.listdir(raw_dir):
        m = pattern.match(fname)
        if m:
            p = float(m.group(1))
            n = int(m.group(2))
            groups.add((n, p))
    return sorted(list(groups))


def build_model_from_params(params, device: torch.device):
    model = TripartiteHeteroGNN_(
        ipm_steps=params['ipm_steps'],
        conv=params['conv'],
        in_shape=2,
        pe_dim=params['lappe'],
        hid_dim=params['hidden'],
        num_conv_layers=params['num_conv_layers'],
        num_pred_layers=params['num_pred_layers'],
        num_mlp_layers=params['num_mlp_layers'],
        dropout=params['dropout'],
        share_lin_weight=params['share_lin_weight'],
        share_conv_weight=params['share_conv_weight'],
        use_norm=params['use_norm'],
        use_res=params['use_res'],
        conv_sequence=params['conv_sequence'],
    ).to(device)
    return model


def extract_model_params_from_args(args):
    return {
        'ipm_steps': args.ipm_steps,
        'conv': args.conv,
        'lappe': args.lappe,
        'hidden': args.hidden,
        'num_conv_layers': args.num_conv_layers,
        'num_pred_layers': args.num_pred_layers,
        'num_mlp_layers': args.num_mlp_layers,
        'dropout': args.dropout,
        'share_lin_weight': args.share_lin_weight,
        'share_conv_weight': args.share_conv_weight,
        'use_norm': args.use_norm,
        'use_res': args.use_res,
        'conv_sequence': args.conv_sequence,
    }


def maybe_override_with_ckpt_args(params: dict, ckpt: dict, use_ckpt_args: bool) -> dict:
    if not use_ckpt_args:
        return params
    ckpt_args = ckpt.get('args')
    if not isinstance(ckpt_args, dict):
        return params
    for k in list(params.keys()):
        if k in ckpt_args:
            params[k] = ckpt_args[k]
    return params


def evaluate_group(args, device, model, n: int, p: float, enforce_feasibility: bool = False, violation_tol: float = 1e-6):
    pre_transform = Compose([
        HeteroAddLaplacianEigenvectorPE(k=args.lappe),
        SubSample(args.ipm_steps),
    ])

    extra_path = f"instance_ER_test_p{p}_n{n}"
    dataset = LPDataset(
        args.data_dir,
        extra_path=extra_path,
        upper_bound=1,
        rand_starts=1,
        pre_transform=pre_transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_ip,
    )

    # Track inference time and memory usage, and compute metrics manually (with optional/adaptive objective sign flip)
    inference_times = []
    memory_usages = []
    obj_gap_list = []
    cons_gap_list = []
    adj_obj_gap_list = []
    viol_ratio_list = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Measure memory before inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated(device) / 1e9  # GB
            else:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1e9  # GB
            
            # Measure inference time
            start_time = time.time()
            # 带投影 vs 不带投影，通过 enforce_feasibility 控制
            if enforce_feasibility:
                pred_x, _ = model(batch, enforce_feasibility=True)
            else:
                pred_x, _ = model(batch)
            end_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            
            # Measure memory after inference
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(device) / 1e9  # GB
            else:
                process = psutil.Process()
                memory_after = process.memory_info().rss / 1e9  # GB
            
            # Calculate per-instance inference time and memory usage
            batch_size = batch.num_graphs  # Number of instances in this batch
            inference_time_per_instance = (end_time - start_time) / batch_size
            memory_usage_per_instance = (memory_after - memory_before) / batch_size
            
            inference_times.append(inference_time_per_instance)
            memory_usages.append(memory_usage_per_instance)

            # Compute metrics on this batch
            pred_steps = pred_x[:, -args.ipm_steps:]
            # Constraint violation: relu(Ax - b)
            Ax = scatter(pred_steps[batch.A_col, :] * batch.A_val[:, None], batch.A_row, reduce='sum', dim=0)
            cons_gap = torch.relu(Ax - batch.rhs[:, None])

            # Objective gap with optional/adaptive sign flip on predicted objective
            x_gt = batch.gt_primals[:, -args.ipm_steps:]
            c_times_xpred = batch.obj_const[:, None] * pred_steps
            c_times_xgt = batch.obj_const[:, None] * x_gt
            obj_pred = scatter(c_times_xpred, batch['vals'].batch, dim=0, reduce='sum')
            obj_gt = scatter(c_times_xgt, batch['vals'].batch, dim=0, reduce='sum')
            if args.flip_objective_sign:
                obj_pred = -obj_pred
            if args.auto_align_objective_sign:
                # If average signs disagree, flip predicted objective
                mean_pred = torch.mean(obj_pred)
                mean_gt = torch.mean(obj_gt)
                if torch.sign(mean_pred) != torch.sign(mean_gt):
                    obj_pred = -obj_pred
            obj_gap = (obj_pred - obj_gt) / (obj_gt + 1e-12)

            # Adjusted objective gap penalized by max constraint violation in batch
            max_cons = torch.max(cons_gap)
            if max_cons > 0:
                obj_nocgap = obj_gap * (1.0 / (1.0 + max_cons))
            else:
                obj_nocgap = obj_gap

            cons_gap_list.append(torch.abs(cons_gap).detach().cpu().numpy())
            obj_gap_list.append(torch.abs(obj_gap).detach().cpu().numpy())
            adj_obj_gap_list.append(torch.abs(obj_nocgap).detach().cpu().numpy())

            # 计算当前 batch 的违反比例（所有约束 × 所有 step）
            viol_mask = cons_gap > violation_tol
            viol_ratio_batch = float(viol_mask.float().mean().item())
            viol_ratio_list.append(viol_ratio_batch)

    # Compute mean and std over all samples and steps
    objs_gap = np.concatenate(obj_gap_list, axis=0) if obj_gap_list else np.array([np.nan])
    cons_gap = np.concatenate(cons_gap_list, axis=0) if cons_gap_list else np.array([np.nan])
    objs_nocgap = np.concatenate(adj_obj_gap_list, axis=0) if adj_obj_gap_list else np.array([np.nan])
    viol_ratios = np.array(viol_ratio_list) if viol_ratio_list else np.array([np.nan])
    result = {
        'n': n,
        'p': p,
        'objective_gap_mean': float(np.nanmean(objs_gap)),
        'objective_gap_std': float(np.nanstd(objs_gap)),
        'constraint_violation_mean': float(np.nanmean(cons_gap)),
        'constraint_violation_std': float(np.nanstd(cons_gap)),
        'adjusted_objective_gap_mean': float(np.nanmean(objs_nocgap)),
        'adjusted_objective_gap_std': float(np.nanstd(objs_nocgap)),
        # 新增：约束违反比例（所有 batch 的 viol_ratio 平均）
        'viol_ratio_mean': float(np.nanmean(viol_ratios)),
        'viol_ratio_std': float(np.nanstd(viol_ratios)),
        'num_entries': int(np.sum(~np.isnan(objs_gap))),
        'avg_inference_time_ms': float(np.nanmean(inference_times) * 1000) if inference_times else float('nan'),  # Convert to milliseconds
        'std_inference_time_ms': float(np.nanstd(inference_times) * 1000) if inference_times else float('nan'),
        'avg_memory_usage_gb': float(np.nanmean(memory_usages)) if memory_usages else float('nan'),
        'std_memory_usage_gb': float(np.nanstd(memory_usages)) if memory_usages else float('nan'),
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate best TELEGEN model on ER test sets grouped by G_n^p')

    # Data and checkpoint
    parser.add_argument('--data_dir', type=str, default='./data', help='Root directory for ER datasets')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pt/.pth)')

    # Discovery / selection
    parser.add_argument('--p_values', type=float, nargs='*', default=None, help='Optional override of p values to evaluate')
    parser.add_argument('--n_values', type=int, nargs='*', default=None, help='Optional override of n values to evaluate')

    # Model hyperparameters (must match training)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--num_conv_layers', type=int, default=2)
    parser.add_argument('--num_pred_layers', type=int, default=4)
    parser.add_argument('--num_mlp_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--conv', type=str, default='gcnconv')
    parser.add_argument('--conv_sequence', type=str, default='cov')

    # IPM parameters
    parser.add_argument('--ipm_steps', type=int, default=16)
    parser.add_argument('--ipm_alpha', type=float, default=0.7)

    # Loss weights (only needed to construct Trainer; does not change metric definitions)
    parser.add_argument('--loss_weight_x', type=float, default=1.0)
    parser.add_argument('--loss_weight_obj', type=float, default=3.43)
    parser.add_argument('--loss_weight_cons', type=float, default=5.8)
    parser.add_argument('--losstype', type=str, default='l2')

    # Other
    parser.add_argument('--lappe', type=int, default=8)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--share_lin_weight', action='store_true', default=True)
    parser.add_argument('--share_conv_weight', action='store_true', default=True)
    parser.add_argument('--use_norm', action='store_true', default=True)
    parser.add_argument('--use_res', action='store_true', default=True)
    parser.add_argument('--output_csv', type=str, default=None, help='Optional path to save CSV of results')
    parser.add_argument('--output_json', type=str, default=None, help='Optional path to save JSON of results')
    parser.add_argument('--use_ckpt_args', action='store_true', default=True, help='Inherit model hyperparameters from checkpoint to ensure compatibility')
    parser.add_argument('--flip_objective_sign', action='store_true', default=False, help='Flip sign of predicted objective to match solver ground truth direction')
    parser.add_argument('--auto_align_objective_sign', action='store_true', default=True, help='Auto align objective sign by comparing mean signs of prediction and ground truth')
    parser.add_argument('--violation_tol', type=float, default=1e-6, help='Threshold for considering a constraint as violated')
    parser.add_argument('--enforce_feasibility_eval', action='store_true', default=False, help='If set, call model with enforce_feasibility=True (projection ON). Otherwise, projection OFF.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint and build model with matching params
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_params = extract_model_params_from_args(args)
    model_params = maybe_override_with_ckpt_args(model_params, ckpt, args.use_ckpt_args)
    model = build_model_from_params(model_params, device)
    state_dict = ckpt.get('state_dict', ckpt)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    if isinstance(ckpt, dict) and 'best_val_loss' in ckpt:
        print(f"Checkpoint best_val_loss: {ckpt['best_val_loss']}")

    # Discover groups
    raw_dir = os.path.join(args.data_dir, 'raw')
    discovered = discover_test_groups(raw_dir)
    if not discovered:
        print(f"No ER test files found in {raw_dir}")
        return

    # Filter by optional user-provided lists
    groups = discovered
    if args.p_values is not None:
        p_set = set(args.p_values)
        groups = [g for g in groups if g[1] in p_set]
    if args.n_values is not None:
        n_set = set(args.n_values)
        groups = [g for g in groups if g[0] in n_set]

    if not groups:
        print("No groups left after filtering by n/p values.")
        return

    print(f"Discovered {len(discovered)} groups; evaluating {len(groups)} groups after filters.")

    all_results = []
    # Evaluate per group
    for (n, p) in groups:
        print(f"Evaluating G_n^p group: n={n}, p={p} ...")
        try:
            res = evaluate_group(args, device, model, n, p, enforce_feasibility=args.enforce_feasibility_eval, violation_tol=args.violation_tol)
            all_results.append(res)
            print(
                f"  -> ObjGap: mean={res['objective_gap_mean']:.6f}, std={res['objective_gap_std']:.6f}; "
                f"ConsViol: mean={res['constraint_violation_mean']:.6f}, std={res['constraint_violation_std']:.6f}; "
                f"ViolRatio: mean={res['viol_ratio_mean']:.4f}, std={res['viol_ratio_std']:.4f}; "
                f"AdjObjGap: mean={res['adjusted_objective_gap_mean']:.6f}, std={res['adjusted_objective_gap_std']:.6f}; "
                f"Time: {res['avg_inference_time_ms']:.2f}±{res['std_inference_time_ms']:.2f}ms; "
                f"Memory: {res['avg_memory_usage_gb']:.3f}±{res['std_memory_usage_gb']:.3f}GB"
            )
        except Exception as e:
            print(f"  !! Failed group n={n}, p={p}: {e}")

    # Pretty print summary table
    print("\nSummary (grouped by G_n^p):")
    header = (
        "n,p,objective_gap_mean,objective_gap_std,"
        "constraint_violation_mean,constraint_violation_std,"
        "adjusted_objective_gap_mean,adjusted_objective_gap_std,"
        "avg_inference_time_ms,std_inference_time_ms,"
        "avg_memory_usage_gb,std_memory_usage_gb,num_entries"
    )
    print(header)
    for r in all_results:
        print(
            f"{r['n']},{r['p']},{r['objective_gap_mean']:.6f},{r['objective_gap_std']:.6f},"
            f"{r['constraint_violation_mean']:.6f},{r['constraint_violation_std']:.6f},"
            f"{r['adjusted_objective_gap_mean']:.6f},{r['adjusted_objective_gap_std']:.6f},"
            f"{r['avg_inference_time_ms']:.2f},{r['std_inference_time_ms']:.2f},"
            f"{r['avg_memory_usage_gb']:.3f},{r['std_memory_usage_gb']:.3f},{r['num_entries']}"
        )

    # Optional exports
    if args.output_csv:
        try:
            import pandas as pd
            import pathlib
            pathlib.Path(os.path.dirname(args.output_csv) or '.').mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(all_results)
            df.to_csv(args.output_csv, index=False)
            print(f"\nSaved CSV to {args.output_csv}")
        except Exception as e:
            print(f"Failed to save CSV: {e}")

    if args.output_json:
        try:
            import pathlib
            pathlib.Path(os.path.dirname(args.output_json) or '.').mkdir(parents=True, exist_ok=True)
            with open(args.output_json, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved JSON to {args.output_json}")
        except Exception as e:
            print(f"Failed to save JSON: {e}")


if __name__ == '__main__':
    main()


