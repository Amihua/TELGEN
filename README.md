# TELGEN

A lightweight, self-contained release of the TELGEN pipeline: a heterogeneous
GNN that learns to approximate the solution of a multi-commodity network-flow
LP (traffic engineering) on Erdos-Renyi (ER) random graphs, trained via
imitation of an interior-point-method (IPM) solver trajectory.

This repo contains just the generate → train → evaluate path for the ER
benchmark: no experiment logs, no alternative topologies (Waxman/B4/Abilene),
no DPO/RLHF variants, no notebooks. If you need those, see the full research
repo this was extracted from.

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
generate_connected_er_data.py   # generates raw ER LP instances (.pkl.gz)
run_er_single_experiment.py     # train + validate + test in one run
eval_er_best_model.py           # evaluate a checkpoint across (n, p) test groups
trainer.py                      # Trainer: objective gap / constraint violation / adjusted objective gap
models/                         # TripartiteHeteroGNN_ and its GNN layers
data/                           # LPDataset, preprocessing transforms, collate utils
solver/                         # LP interior-point solver used both for data generation and IPM supervision
```

## Install

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.3, PyTorch Geometric 2.5. `torch-scatter`
and `torch-sparse` are **not required**: every place that used them has a
pure-PyTorch fallback (see "Notes" below), so a missing or ABI-mismatched
build of those packages will not break anything.

## 1. Generate data

```bash
python generate_connected_er_data.py --root_dir ./data/raw
```

Defaults produce the standard benchmark grid: training graphs with
`n in [20..100]` (step 10) x `p in [0.3..0.8]`, 200 connected instances
each; test graphs with `n in {200, 500, 1000, 2000}` x
`p in [0.1..0.9]`, 300 instances each. Every generated graph is checked
for connectivity (regenerated with a new seed if disconnected) before the
LP is built and solved. Use `--smoke` for a tiny sanity-check run, or
`--only train` / `--only test` to generate one half of the grid. See
`--help` for capacity/demand ranges, SD pair count, and k-shortest-path k.

Files are written as `instance_ER_{train,test}_p{p}_n{n}.pkl.gz` under
`--root_dir` (default `./data/raw`).

## 2. Train

```bash
python run_er_single_experiment.py \
  --data_dir ./data \
  --test_p 0.9 --test_n 2000 \
  --epochs 50 --lr 5e-4 --batchsize 32
```

This concatenates **all** `instance_ER_train_*.pkl.gz` files under
`./data/raw` into one training set, carves out a validation split
(`--val_seed` / `--val_size`) from it, trains for `--epochs`, and reports
final objective gap / constraint violation / adjusted objective gap on
the `(--test_p, --test_n)` test set. The best checkpoint (by validation
loss) is saved to `--ckpt_dir` (default `./checkpoints`) as
`best_model.pth`, alongside the args needed to rebuild the model.

## 3. Evaluate a checkpoint across densities

```bash
python eval_er_best_model.py \
  --checkpoint ./checkpoints/<run_name>/best_model.pth \
  --data_dir ./data \
  --output_csv results.csv
```

Scans `./data/raw` for every available `(n, p)` test group, evaluates the
checkpoint on each, and prints/saves a table of objective gap, constraint
violation, and adjusted objective gap (mean ± std) per group. Restrict to
specific groups with `--p_values` / `--n_values`.

## Notes

- **BatchNorm needs batch size > 1.** With very small datasets (e.g. a
  `--smoke` run), pick a `--batchsize` that avoids a trailing batch of
  size 1, or the last batch of an epoch will raise a `ValueError` from
  `torch.nn.functional.batch_norm`. This is a property of the model
  (`use_norm=True` uses BatchNorm1d in the encoder), not a data-loading bug.
- **torch-scatter / torch-sparse fallbacks.** `trainer.py` and
  `data/dataset.py` try to import `torch_scatter.scatter` and
  `torch_sparse.SparseTensor` respectively; if either import fails (e.g.
  an ABI mismatch against your installed PyTorch build), both fall back to
  a small pure-PyTorch equivalent (`scatter_add`-based) so the pipeline
  still runs — just without the compiled kernels' speed.

## License

Apache-2.0, see [LICENSE](LICENSE).
