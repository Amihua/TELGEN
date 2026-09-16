# ASN98 reproduction experiment

## TL;DR

The original ASN pipeline (`gen_reallocation_new_train_test_asn_graph_sampling.ipynb`)
samples train/test node sets **uniformly at random** from the 1739-node `ASN2k.json`
backbone. Its leftover `train_98_nodes.npy` trains very poorly: validation objgap
plateaus at 15-25% regardless of learning rate, epoch count, model capacity (`hidden`,
`num_conv_layers`), `ipm_alpha`, constraint-loss weight, or training-set size.
## Contents

```
raw/asn_graph/ASN2k.json        # full ASN backbone
raw/asn_graph/dense98_nodes.npy # the 98-node subgraph actually used
raw_asn_saved/                  # generated LP instances
  instance_train_dense98.pkl.gz #   400 instances (seed=2024)
  instance_test_dense98.pkl.gz  #   150 instances (seed=2025, same topology)
raw/processed_..._train_dense98/  # LPDataset cache (IPM supervision trajectories included)
raw/processed_..._test_dense98/   #   -- both load straight from here, no reprocessing needed
checkpoints/best_model.pt       # trained model (see Results below)
gen_asn_dense98.ipynb           # documents the density search + generates the LP instances
run_reallocation.ipynb          # train + test (edited copy of the original notebook, see below)
data/ models/ solver/ trainer.py           # supporting library code
```


## Environment

Needs `torch_geometric`, `torch_scatter`, `torch_sparse` (real builds). Tested with the
`GAS` conda env on this machine (`torch_geometric==2.7`, PyTorch 2.6/cu124). PyTorch 2.6
defaults `torch.load(weights_only=True)`, which breaks loading the cached PyG
Batch/HeteroData objects -- `data/dataset.py` here already has `weights_only=False`
patched into its `torch.load` calls.

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
test_ds = LPDataset('raw', extra_path=f'1restarts_0lap_{ipm}steps_upper_test_dense98',
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
| this checkpoint (ASN98, latest results) | 7.83% | 2.87% | **1.46%** |
| paper Table II, ASN98 (TELGEN) | 2.31% | 0.63% | **1.70%** |

`OnoCGap` is the trainer's own definition: `OGap * 1/(1+max violation across the whole
eval batch)`. 


## Other configs tried (same dense ASN98 topology)

| variant | train set | epochs | wcons | OGap | CGap | OnoCGap |
|---|---|---|---|---|---|---|
| **shipped checkpoint** | 400 | 800 | 5.8 | 7.83% | 2.87% | **1.46%** |
| bigger data + higher wcons | 1500 | 500 | 10 | 8.43% | 2.93% | 1.69% |
| shorter run | 400 | 300 | 5.8 | 8.92% | 2.77% | 1.65% |

All land in the same range: **OGap 7.83-8.92%, CGap 2.77-2.93%, OnoCGap 1.46-1.69%**.
A couple of denser variants (a denser 98-node subgraph, and the same recipe scaled to a
237-node subgraph) were also tried and landed slightly worse (OnoCGap 2.2-2.5%) — not
shipped here since the topology above was the best found; a finer sweep around it was
not run.

## Regenerating from scratch

`raw_asn_saved/*.pkl.gz` and the `processed_*/` caches are already built — you don't need
to redo this. If you want to anyway:

1. Run `gen_asn_dense98.ipynb`. It writes `raw/raw/instance_0.pkl.gz`, **overwritten**
   between train/test generation calls — the notebook copies each split out to
   `raw_asn_saved/` right after generating it, same pattern as `b4_exp/`.
2. `LPDataset('raw', extra_path='1restarts_0lap_16steps_upper_train_dense98', ...)` (and
   `..._test_dense98`) build the processed cache from whatever is in `raw/raw/*.pkl.gz`
   at call time — the notebook already does this right after each generation step.
3. `run_reallocation.ipynb` trains + evaluates from the cache built above.
