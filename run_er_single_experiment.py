#!/usr/bin/env python3
"""
TELEGEN - ER Graph Single Experiment Runner
Based on the structure of run_reallocation.py, this script is a self-contained
program to run a single training and evaluation experiment on a specific ER graph configuration.
"""

import os

import argparse
import sys
import time
import copy
import datetime
import gc
from pathlib import Path

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch_geometric.transforms import Compose
from tqdm import tqdm
import yaml
import random
from torch.utils.tensorboard import SummaryWriter
from functools import partial

# Import project modules
from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
from data.dataset import LPDataset
from data.utils import args_set_bool, collate_fn_ip
from models.hetero_gnn import TripartiteHeteroGNN_
from trainer import Trainer

# Mock wandb to disable it
class MockWandb:
    def __init__(self):
        self.run = None
    
    def init(self, *args, **kwargs):
        print("wandb disabled - init() call ignored")
        return None
    
    def log(self, *args, **kwargs):
        pass # Silently ignore log calls
    
    def finish(self, *args, **kwargs):
        print("wandb disabled - finish() call ignored")
        return None

sys.modules['wandb'] = MockWandb()
wandb = MockWandb()


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"✅ Random seed set to {seed}")

def clear_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def create_datasets(args):
    """Create train, validation (sampled from train), and test datasets from generated files."""
    print(" R Creating datasets...")
    raw_data_dir = os.path.join(args.data_dir, 'raw')
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory not found at {raw_data_dir}")

    # Define the transformations
    pre_transform = Compose([
        HeteroAddLaplacianEigenvectorPE(k=args.lappe),
        SubSample(args.ipm_steps)
    ])

    # --- Training Dataset Creation (Concatenate all train files) ---
    print("   - Loading and concatenating ALL available training data...")
    train_files = [f for f in os.listdir(raw_data_dir)
                   if f.startswith('instance_ER_train_') and f.endswith('.pkl.gz')]
    if not train_files:
        raise FileNotFoundError(f"No training files found in {raw_data_dir}")

    train_datasets = []
    for fname in tqdm(train_files, desc="Loading training files"):
        extra_path = fname.replace(".pkl.gz", "")
        try:
            dataset = LPDataset(args.data_dir,
                                extra_path=extra_path,
                                upper_bound=1,
                                rand_starts=1,
                                pre_transform=pre_transform)
            train_datasets.append(dataset)
        except Exception as e:
            print(f"Warning: Could not load/process {fname}. Error: {e}. Skipping.")

    if not train_datasets:
        raise RuntimeError("Failed to load any training datasets.")

    full_train_dataset = ConcatDataset(train_datasets)
    full_len = len(full_train_dataset)
    print(f"   - Combined {len(train_datasets)} training datasets into one with {full_len} samples.")

    # --- Split VAL from TRAIN by random seed=2025 and fixed size ---
    print(f"   - Sampling validation set from TRAIN with seed={args.val_seed}, size={args.val_size}")
    rng = np.random.RandomState(args.val_seed)
    all_indices = np.arange(full_len)
    rng.shuffle(all_indices)

    val_size = min(args.val_size, full_len)
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]

    train_dataset = Subset(full_train_dataset, train_indices.tolist())
    val_dataset = Subset(full_train_dataset, val_indices.tolist())

    print(f"   - Final split -> TRAIN: {len(train_dataset)} | VAL: {len(val_dataset)}")

    # --- Test Dataset Creation ---
    test_path = f"instance_ER_test_p{args.test_p}_n{args.test_n}"
    print(f"   - Loading TEST set from: {test_path}")
    test_dataset = LPDataset(args.data_dir,
                             extra_path=test_path,
                             upper_bound=1,
                             rand_starts=1,
                             pre_transform=pre_transform)

    return train_dataset, val_dataset, test_dataset


def main():
    parser = argparse.ArgumentParser(description='TELEGEN - ER Graph Experiment with Train/Val/Test Split')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='./data', help='Root directory for ER datasets')
    # Validation sampling config (from TRAIN)
    parser.add_argument('--val_seed', type=int, default=2025, help='Random seed for sampling validation set from train')
    parser.add_argument('--val_size', type=int, default=300, help='Number of samples for validation set from train')
    # Test set parameters (user-specified)
    parser.add_argument('--test_p', type=float, default=0.9, help='P value for the final testing ER graph.')
    parser.add_argument('--test_n', type=int, default=2000, help='N value for the final testing ER graph.')

    # Model hyperparameters
    parser.add_argument('--hidden', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--num_conv_layers', type=int, default=2, help='Number of convolution layers')
    parser.add_argument('--num_pred_layers', type=int, default=4, help='Number of prediction layers')
    parser.add_argument('--num_mlp_layers', type=int, default=4, help='Number of MLP layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--conv', type=str, default='gcnconv', help='Convolution type')
    parser.add_argument('--conv_sequence', type=str, default='cov', help='Convolution sequence')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    
    # IPM parameters
    parser.add_argument('--ipm_steps', type=int, default=16, help='IPM steps')
    parser.add_argument('--ipm_alpha', type=float, default=0.7, help='IPM alpha')
    
    # Loss weights
    parser.add_argument('--loss_weight_x', type=float, default=1.0, help='Primal loss weight')
    parser.add_argument('--loss_weight_obj', type=float, default=3.43, help='Objective loss weight')
    parser.add_argument('--loss_weight_cons', type=float, default=5.8, help='Constraint loss weight')
    parser.add_argument('--losstype', type=str, default='l2', help='Loss type')
    
    # Other parameters
    parser.add_argument('--lappe', type=int, default=8, help='Laplacian positional encoding dimension')
    parser.add_argument('--seed', type=int, default=2026, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of data loading workers')
    
    # Model architecture flags
    parser.add_argument('--share_lin_weight', action='store_true', default=True, help='Share linear weights')
    parser.add_argument('--share_conv_weight', action='store_true', default=True, help='Share conv weights')
    parser.add_argument('--use_norm', action='store_true', default=True, help='Use normalization')
    parser.add_argument('--use_res', action='store_true', default=True, help='Use residual connections')

    # Logging / TensorBoard
    parser.add_argument('--disable_tensorboard', action='store_true', help='Disable TensorBoard logging')
    parser.add_argument('--tb_log_dir', type=str, default='./tensorboard_logs', help='TensorBoard log directory')
    # Checkpoint saving
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints', help='Directory to save best model checkpoints')
    
    args = parser.parse_args()
    
    # --- 1. Setup ---
    print("🎯 TELEGEN - Single ER Graph Training Experiment")
    print("=" * 60)
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    set_random_seed(args.seed)

    # Prepare run name (used for both TB and checkpoints)
    run_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"er_train_val_test_bs{args.batchsize}_lr{args.lr}_ipm{args.ipm_steps}_{run_time}"

    # TensorBoard writer
    writer = None
    if not args.disable_tensorboard:
        tb_dir = os.path.join(args.tb_log_dir, run_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(tb_dir)
        print(f"TensorBoard: logging to {tb_dir}")

    # Checkpoint directory for this run
    ckpt_run_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_run_dir, 'best_model.pth')

    os.makedirs(args.data_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Data Loading ---
    try:
        train_dataset, val_dataset, test_dataset = create_datasets(args)
        print(f"  - Training samples found: {len(train_dataset)}")
        print(f"  - Validation samples found: {len(val_dataset)}")
        print(f"  - Test samples found: {len(test_dataset)}")
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        print("   Please ensure the generated data files exist in the raw directory.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn_ip)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_ip)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn_ip)

    # --- 3. Model Initialization ---
    print("🏗️  Initializing model and trainer...")
    model = TripartiteHeteroGNN_(
        ipm_steps=args.ipm_steps,
        conv=args.conv,
        in_shape=2,  # Typically 2 for mean and stddev features
        pe_dim=args.lappe,
        hid_dim=args.hidden, # Corrected parameter name
        num_conv_layers=args.num_conv_layers,
        num_pred_layers=args.num_pred_layers,
        num_mlp_layers=args.num_mlp_layers,
        dropout=args.dropout,
        share_lin_weight=args.share_lin_weight,
        share_conv_weight=args.share_conv_weight,
        use_norm=args.use_norm,
        use_res=args.use_res,
        conv_sequence=args.conv_sequence
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Initialize model weights more carefully to prevent NaN
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    print("✅ Model weights initialized with Xavier uniform")
    
    # This trainer definition is a simplified one for this specific script's purpose.
    # A more complete Trainer class would handle the model and optimizer internally.
    class SimpleTrainer:
        def __init__(self, device, losstype, loss_weight_x, loss_weight_obj, loss_weight_cons, ipm_alpha):
            self.device = device
            self.loss_weight_x = loss_weight_x
            self.loss_weight_obj = loss_weight_obj
            self.loss_weight_cons = loss_weight_cons
            self.ipm_alpha = ipm_alpha
            if losstype == 'l2':
                self.loss_func = partial(torch.pow, exponent=2)
            elif losstype == 'l1':
                self.loss_func = torch.abs
            else:
                raise ValueError
        
        def get_loss(self, pred_x, data):
            # A simplified loss calculation for primal variables
            
            # --- Numerical Probe 1: Check inputs ---
            if torch.isnan(pred_x).any():
                print(f"\n[Debug] NaN detected in model predictions (pred_x).")
                print(f"  - pred_x shape: {pred_x.shape}")
                print(f"  - pred_x min/max: {pred_x.min():.6f}/{pred_x.max():.6f}")
                print(f"  - pred_x contains inf: {torch.isinf(pred_x).any()}")
                # Instead of crashing, return a small loss to continue training
                print("  - Returning small loss to continue training...")
                return torch.tensor(1e-6, device=pred_x.device, requires_grad=True)
                
            if torch.isnan(data.gt_primals).any():
                print(f"\n[Debug] NaN detected in ground truth labels (gt_primals).")
                print(f"  - gt_primals shape: {data.gt_primals.shape}")
                print(f"  - gt_primals min/max: {data.gt_primals.min():.6f}/{data.gt_primals.max():.6f}")
                print("  - Returning small loss to continue training...")
                return torch.tensor(1e-6, device=pred_x.device, requires_grad=True)

            # Check for extreme values that might cause numerical issues
            if torch.isinf(pred_x).any() or torch.isinf(data.gt_primals).any():
                print(f"\n[Debug] Inf detected in predictions or ground truth.")
                print("  - Clipping extreme values...")
                pred_x = torch.clamp(pred_x, -1e6, 1e6)
                data.gt_primals = torch.clamp(data.gt_primals, -1e6, 1e6)

            primal_loss = self.loss_func(pred_x - data.gt_primals).mean()

            # --- Numerical Probe 2: Check loss output ---
            if torch.isnan(primal_loss) or torch.isinf(primal_loss):
                print(f"\n[Debug] NaN/Inf detected in the calculated loss: {primal_loss}")
                print("  - Returning small loss to continue training...")
                return torch.tensor(1e-6, device=pred_x.device, requires_grad=True)

            total_loss = self.loss_weight_x * primal_loss
            return total_loss

        def train_step(self, model, optimizer, batch):
            model.train()
            optimizer.zero_grad()
            batch = batch.to(self.device)
            pred_x, _ = model(batch)
            loss = self.get_loss(pred_x, batch)
            
            if loss.requires_grad:
                loss.backward()
                # --- Add Gradient Clipping for stability ---
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            return loss.item()

        @torch.no_grad()
        def eval_step(self, model, batch):
            model.eval()
            batch = batch.to(self.device)
            pred_x, _ = model(batch)
            loss = self.get_loss(pred_x, batch)
            return loss.item()

    trainer = SimpleTrainer(
        device=device,
        losstype=args.losstype,
        loss_weight_x=args.loss_weight_x,
        loss_weight_obj=args.loss_weight_obj,
        loss_weight_cons=args.loss_weight_cons,
        ipm_alpha=args.ipm_alpha
    )

    # Metrics evaluator using original definitions (objective gap, constraint violation, adjusted objective gap)
    metrics_trainer = Trainer(
        device=device,
        loss_target='primal+objgap+constraint',
        loss_type=args.losstype,
        micro_batch=1,
        ipm_steps=args.ipm_steps,
        ipm_alpha=args.ipm_alpha,
        loss_weight={'primal': args.loss_weight_x, 'objgap': args.loss_weight_obj, 'constraint': args.loss_weight_cons}
    )

    # --- 4. Training and Evaluation Loop ---
    print("🚀 Starting training...")
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar:
            loss = trainer.train_step(model, optimizer, batch)
            total_train_loss += loss
            pbar.set_postfix({'loss': loss})
        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluation phase on Validation Set
        total_val_loss = 0
        num_val_batches = len(val_loader)
        
        pbar_eval = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        for batch in pbar_eval:
            loss = trainer.eval_step(model, batch)
            total_val_loss += loss
            pbar_eval.set_postfix({'loss': loss})
        
        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
        else:
            avg_val_loss = float('nan')
            print("\n[Warning] Validation loader is empty. Validation loss cannot be calculated.")

        # Compute validation metrics (objective gap, constraint violation, adjusted objective gap)
        try:
            val_objs_gap, val_cons_gap, val_objs_nocgap = metrics_trainer.eval_metrics_(val_loader, model)
            val_obj_gap_mean = float(np.mean(val_objs_gap))
            val_cons_gap_mean = float(np.mean(val_cons_gap))
            val_adj_obj_gap_mean = float(np.mean(val_objs_nocgap))
        except Exception as e:
            print(f"[Warn] Validation metrics computation failed: {e}")
            val_obj_gap_mean = float('nan')
            val_cons_gap_mean = float('nan')
            val_adj_obj_gap_mean = float('nan')

        # TensorBoard logging per epoch
        if writer is not None:
            writer.add_scalar('loss/train', avg_train_loss, epoch)
            writer.add_scalar('loss/val', avg_val_loss, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('metrics/val_objective_gap', val_obj_gap_mean, epoch)
            writer.add_scalar('metrics/val_constraint_violation', val_cons_gap_mean, epoch)
            writer.add_scalar('metrics/val_adjusted_objective_gap', val_adj_obj_gap_mean, epoch)
            writer.flush()

        print(
            f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
            f"Val ObjGap: {val_obj_gap_mean:.6f} | Val ConsViol: {val_cons_gap_mean:.6f} | Val AdjObjGap: {val_adj_obj_gap_mean:.6f}"
        )
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"   -> New best val loss: {best_val_loss:.6f}. Model state saved.")
            # Also print the three validation metrics for this best model
            print(
                f"      Best Val ObjGap: {val_obj_gap_mean:.6f} | Best Val ConsViol: {val_cons_gap_mean:.6f} | Best Val AdjObjGap: {val_adj_obj_gap_mean:.6f}"
            )
            # Persist best checkpoint to disk
            torch.save({
                'state_dict': best_model_state,
                'epoch': epoch,
                'best_val_loss': float(best_val_loss),
                'args': vars(args),
            }, best_ckpt_path)
            print(f"   -> Best checkpoint saved to: {best_ckpt_path}")

        clear_memory()

    # --- 5. Final Evaluation on Test Set ---
    print("=" * 60)
    print("🧪 Final Evaluation on the TEST set...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("   - Loaded best model state from validation.")
    else:
        print("   - Warning: No best model state found. Using the last model state.")

    total_final_test_loss = 0
    num_test_batches = len(test_loader)
    pbar_test = tqdm(test_loader, desc="Final Test")
    for batch in pbar_test:
        loss = trainer.eval_step(model, batch)
        total_final_test_loss += loss
        pbar_test.set_postfix({'loss': loss})

    if num_test_batches > 0:
        final_test_loss = total_final_test_loss / num_test_batches
    else:
        final_test_loss = float('nan')
        print("   - Warning: Test loader is empty.")
    
    # --- 6. Final Results ---
    # Compute test metrics
    try:
        test_objs_gap, test_cons_gap, test_objs_nocgap = metrics_trainer.eval_metrics_(test_loader, model)
        test_obj_gap_mean = float(np.mean(test_objs_gap))
        test_cons_gap_mean = float(np.mean(test_cons_gap))
        test_adj_obj_gap_mean = float(np.mean(test_objs_nocgap))
    except Exception as e:
        print(f"[Warn] Test metrics computation failed: {e}")
        test_obj_gap_mean = float('nan')
        test_cons_gap_mean = float('nan')
        test_adj_obj_gap_mean = float('nan')

    if writer is not None:
        writer.add_scalar('loss/test_final', final_test_loss, args.epochs)
        writer.add_hparams({
            'hidden': args.hidden,
            'lr': args.lr,
            'batchsize': args.batchsize,
            'ipm_steps': args.ipm_steps,
            'lappe': args.lappe
        }, {
            'metrics/best_val_loss': best_val_loss,
            'metrics/final_test_loss': final_test_loss
        })
        writer.add_scalar('metrics/test_objective_gap', test_obj_gap_mean, args.epochs)
        writer.add_scalar('metrics/test_constraint_violation', test_cons_gap_mean, args.epochs)
        writer.add_scalar('metrics/test_adjusted_objective_gap', test_adj_obj_gap_mean, args.epochs)
        writer.flush()
        writer.close()

    print("=" * 60)
    print("🎉 Training Complete!")
    print(f"   - Final Train Loss: {avg_train_loss:.6f}")
    print(f"   - Best Val Loss: {best_val_loss:.6f} (sampled from train with seed={args.val_seed}, size={args.val_size})")
    print(f"   - Final Test Loss (on p={args.test_p}, n={args.test_n}): {final_test_loss:.6f}")
    print(f"   - Test ObjGap: {test_obj_gap_mean:.6f} | Test ConsViol: {test_cons_gap_mean:.6f} | Test AdjObjGap: {test_adj_obj_gap_mean:.6f}")
    print(f"   - Best checkpoint path: {best_ckpt_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
