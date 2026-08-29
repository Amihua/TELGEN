# TELGEN：Traffic Engineering in Large-scale Networks with Generalizable Graph
## Paper

This is the code release for:

> **Traffic Engineering in Large-scale Networks with Generalizable Graph
> Neural Networks**
> Fangtong Zhou, Sihao Liu, Xiaorui Liu, Ruozhou Yu, Guoliang Xue.
> *IEEE Transactions on Networking*, 2026.
> [IEEE Xplore](https://ieeexplore.ieee.org/document/11367762/)

If you use this code, please cite:

```bibtex
@article{zhou2026traffic,
  title={Traffic engineering in large-scale networks with generalizable graph neural networks},
  author={Zhou, Fangtong and Liu, Sihao and Liu, Xiaorui and Yu, Ruozhou and Xue, Guoliang},
  journal={IEEE Transactions on Networking},
  year={2026},
  publisher={IEEE}
}
```

## Problem setup

Each LP instance is a multi-commodity flow problem on a connected ER graph:

- **Input**: a graph `G(V, E, C)` with random edge capacities, a set of
  source-destination (s,t) pairs with random demands, and k-shortest
  candidate paths per pair.
- **Objective**: maximize total routed demand.
- **Constraints**: for every commodity, flow across its candidate paths
  sums to at most 1 (normalized); for every edge, total flow must not
  exceed capacity.

The model (`TripartiteHeteroGNN_`) is a heterogeneous GNN over a
tripartite graph of constraint / variable / objective nodes, trained to
imitate the intermediate iterates of a SciPy interior-point LP solver
(`solver/linprog.py`, adapted from `scipy.optimize.linprog`) across
`ipm_steps` steps.

## Repo layout

```
demo.py                         # run the shipped model on the large (n=1000/2000) test graphs
generate_connected_er_data.py   # generates raw ER LP instances (.pkl.gz)
run_er_single_experiment.py     # train + validate + test in one run
train_er_weighted.py            # train with the full weighted loss (primal + objgap + constraint)
eval_er_best_model.py           # evaluate a checkpoint across (n, p) test groups (IPM-iterate baseline)
eval_er_gurobi_metric.py        # evaluate against the TRUE LP optimum (Gurobi) with the paper's metrics
trainer.py                      # Trainer: objective gap / constraint violation / adjusted objective gap
models/                         # TripartiteHeteroGNN_, its GNN layers, and the feasibility projection layer
data/                           # LPDataset, preprocessing transforms, collate utils
solver/                         # LP interior-point solver used both for data generation and IPM supervision
checkpoints/best_model.pth      # a trained model, ready to use with demo.py / eval_er_gurobi_metric.py
data/raw/                       # bundled ER instances: 54 train files (p 0.3-0.8) + 36 test files
```

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.3, PyTorch Geometric 2.5. `torch-scatter`
and `torch-sparse` are **not required**: every place that used them has a
pure-PyTorch fallback (see "Notes" below), so a missing or ABI-mismatched
build of those packages will not break anything.

`gurobipy` is optional — used only to compute the exact LP optimum in `demo.py`
and `eval_er_gurobi_metric.py`. Without it, the bundled interior-point solver in
`solver/` is used instead (same objective value on these LPs).

## Quick demo

The repo ships a trained model (`checkpoints/best_model.pth`) and the ER test
instances (`data/raw/instance_ER_test_*.pkl.gz`), so you can run it right away —
no data generation or training needed:

```bash
python demo.py                       # ER n=1000, p=0.5, 50 instances
python demo.py --n 2000 --p 0.9      # a larger / denser test set
python demo.py --n 1000 --p 0.3 --num_instances 100
python demo.py --list                # list every available (n, p) test set
```

`demo.py` runs the model on the **large** ER graphs (`n = 1000` and `n = 2000`),
which are 10-20x bigger than anything in training, and prints:

| column  | meaning |
|---------|---------|
| OGap    | objective (optimality) gap of the raw model output vs the LP optimum |
| CGap    | total constraint violation of the raw model output (paper's `γ_con`) |
| OnoCGap | objective gap after a cheap per-link feasibility restoration |
| time    | model forward time per instance (single, un-batched) |

Example output (`--n 1000 --p 0.5`):

```
               OGap     CGap   OnoCGap   time/inst
  TELGEN      0.05%    2.41%     0.14%    ...
```


## 1. Generate data

```bash
python generate_connected_er_data.py --root_dir ./data/raw
```


## 2. Train

```bash
python run_er_single_experiment.py \
  --data_dir ./data \
  --test_p 0.9 --test_n 2000 \
  --epochs 50 --lr 5e-4 --batchsize 32
```


## 3. Evaluate a checkpoint across densities

```bash
python eval_er_best_model.py \
  --checkpoint ./checkpoints/<run_name>/best_model.pth \
  --data_dir ./data \
  --output_csv results.csv
```




## License

Apache-2.0, see [LICENSE](LICENSE).
