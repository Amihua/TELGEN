# B4 reproduction experiment

## Contents

```
raw/b4_graph/                  # B4 topology: B4.json + train/test node subsets
raw_b4_saved/                  # generated LP instances (seed=2024)
  instance_train_b4.pkl.gz     #   200 instances
  instance_test_b4.pkl.gz      #   100 instances
raw/processed_..._train_b4/    # LPDataset cache (IPM supervision trajectories included)
raw/processed_..._test_b4/     #   -- both load straight from here, no reprocessing needed
checkpoints/best_model.pt      # trained model (see Results below)
gen_reallocation_new_train_test_b4.ipynb   # generates the LP instances from raw/b4_graph/
run_reallocation.ipynb         # train + test (original notebook, see caveat below)
data/ models/ solver/ trainer.py           # supporting library code
```

## Environment

Needs `torch_geometric`, `torch_scatter`, `torch_sparse` (real builds, not the
pure-PyTorch fallback the top-level release repo uses). Tested with the `GAS`
conda env on this machine (`torch_geometric==2.7`, PyTorch 2.6/cu124).

## Quick eval (use the shipped checkpoint, no training)

```python
import torch
from torch_geometric.transforms import Compose
from data.data_preprocess import HeteroAddLaplacianEigenvectorPE, SubSample
from data.dataset import LPDataset
from models.hetero_gnn import TripartiteHeteroGNN_
from trainer import Trainer

ipm = 16
pre = Compose([HeteroAddLaplacianEigenvectorPE(k=0), SubSample(ipm)])
test_ds = LPDataset('raw', extra_path=f'1restarts_0lap_{ipm}steps_upper_test_b4',
                     upper_bound=1, rand_starts=1, pre_transform=pre)   # loads from the cache above

model = TripartiteHeteroGNN_(ipm_steps=ipm, conv='gcnconv', in_shape=2, pe_dim=0, hid_dim=180,
    num_conv_layers=2, num_pred_layers=4, num_mlp_layers=4, dropout=0.0, share_conv_weight=True,
    share_lin_weight=True, use_norm=True, use_res=True, conv_sequence='cov')
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.eval()
```

## Results (`checkpoints/best_model.pt`, test set, `trainer.eval_metrics_`)

| | OGap | CGap | OnoCGap |
|---|---|---|---|
| this checkpoint | 5.94% | 1.62% | 1.75% |
| paper Table II, B4 (TELGEN) | 6.40% | 3.97% | 2.99% |

`OnoCGap` here is the plain definition (`OGap * 1/(1+max violation)`, one global
rescale)

## Regenerating from scratch

`raw_b4_saved/*.pkl.gz` and the `processed_*/` caches are already built (seed=2024,
both splits) -- you don't need to redo this. If you want to anyway:

1. Run `gen_reallocation_new_train_test_b4.ipynb`. It writes `raw/raw/instance_0.pkl.gz`,
   **overwritten** between train/test runs -- copy each split out (as `raw_b4_saved/`
   does here) before generating the other.
2. `LPDataset('raw', extra_path='1restarts_0lap_16steps_upper_train_b4', ...)` (and
   `..._test_b4`) build the processed cache from whatever is in `raw/raw/*.pkl.gz` at
   call time -- so cache `train_b4` right after generating train, before test
   overwrites `raw/raw/`.




