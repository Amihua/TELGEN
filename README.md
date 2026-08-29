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

## Reproduce our results with the shipped checkpoint

Everything needed to reproduce the ER evaluation tables is **in this
repository** — no external downloads. `git clone` gives you the code, the
54 training + 36 test instance files (`data/raw/`, ~13 MB), and one trained
model (`checkpoints/best_model.pth`, 1.1 MB, a plain git blob — **not** Git
LFS, so a normal clone is enough).

### 1. Environment

```bash
# Python 3.10 or 3.12 (both tested); a fresh venv/conda env is recommended
python -m venv .venv && source .venv/bin/activate      # or: conda create -n telgen python=3.10

pip install -r requirements.txt
```

This pulls `torch>=2.1`, `torch-geometric>=2.5`, `numpy`, `scipy`,
`networkx`, `tqdm`, `psutil`. Tested with PyTorch 2.3 / PyG 2.5 on both CPU
and a single CUDA GPU.

- `torch-scatter` / `torch-sparse` are **not required** — every path that
  used them has a pure-PyTorch fallback, so a missing or ABI-mismatched
  build will not break anything.
- **Gurobi is optional.** `demo.py` and `eval_er_gurobi_metric.py` use it to
  get the exact LP optimum. These LPs are tiny (~40 variables, ~90
  constraints), so Gurobi's **free restricted license is sufficient**:
  `pip install gurobipy`. Without Gurobi the bundled interior-point solver
  in `solver/` is used instead and produces the same objective values.

### 2. Data

Already in place at `data/raw/` after cloning:

```
data/raw/instance_ER_train_p{0.3..0.8}_n{20..100}.pkl.gz   # 54 files, training
data/raw/instance_ER_test_p{0.1..0.9}_n{200,500,1000,2000}.pkl.gz   # 36 files, test
```

Each `.pkl.gz` is a list of `(A, b, c)` LP tuples. Nothing to download or
move. (You can regenerate them from scratch with
`python generate_connected_er_data.py --root_dir ./data/raw` — see §5 —
but the committed files are what the numbers below were produced from.)

### 3. Checkpoint

Already in place at `checkpoints/best_model.pth` after cloning. Config
(read back from the file): `hidden=64`, `lappe=8`, `ipm_steps=16`,
`ipm_alpha=0.7`, `losstype=l1`, `conv=gcnconv`, trained 150 epochs with the
weighted loss (`train_er_weighted.py`), best epoch selected on validation
objective gap. All scripts default `--checkpoint checkpoints/best_model.pth`.

### 4. Evaluate

All RNGs (`torch`, `numpy`, cuDNN, Gurobi, the Laplacian-PE eigensolver) are
pinned, so the metric columns are **byte-reproducible** across runs on the
same machine; `--seed 0` is the default. Small run-to-run differences
between machines come only from BLAS/Gurobi floating-point order.

#### 4a. Quick check — `demo.py` (large graphs, ~1 min)

`demo.py` runs the model on the **large** ER graphs (`n = 1000`, `2000`),
10-20x bigger than anything in training, and reports:

| column  | meaning |
|---------|---------|
| OGap    | objective (optimality) gap of the raw model output vs the LP optimum |
| CGap    | total constraint violation of the raw model output (paper's `γ_con`) |
| OnoCGap | objective gap after a cheap per-link feasibility restoration |
| time    | model forward time per instance (single, un-batched; machine-dependent) |

```bash
python demo.py --n 1000 --p 0.5           # 50 instances (default)
python demo.py --n 2000 --p 0.9
python demo.py --list                     # list every available (n, p) test set
```

Expected output (this checkpoint, `--seed 0`, 50 instances):

```
# python demo.py --n 1000 --p 0.5
               OGap     CGap   OnoCGap   time/inst
  TELGEN      0.07%    3.30%     0.19%     ~330ms
  paper       3.34%    5.21%     0.48%

# python demo.py --n 2000 --p 0.9
  TELGEN      0.05%    1.67%     0.10%     ~340ms
  paper       3.18%    5.48%     0.58%
```

(The `paper` row is the published Table V value for that `(n, p)`, printed
for side-by-side reference. `time/inst` is machine-dependent.)

#### 4b. Full ER gap table (paper Table V) — `eval_er_gurobi_metric.py`

```bash
python eval_er_gurobi_metric.py \
  --checkpoint checkpoints/best_model.pth \
  --raw_dir data/raw \
  --n_values 200 500 1000 2000 \
  --p_values 0.1 0.5 0.9 \
  --max_instances 100 \
  --seed 0 \
  --output_csv table5_repro.csv
```



## Regenerate the data (optional)

```bash
python generate_connected_er_data.py --root_dir ./data/raw
```

## Train from scratch (optional)

```bash
# simple single-run trainer
python run_er_single_experiment.py \
  --data_dir ./data --test_p 0.9 --test_n 2000 \
  --epochs 50 --lr 5e-4 --batchsize 32

# full weighted loss (primal + objective-gap + constraint), as used for the shipped checkpoint
python train_er_weighted.py \
  --data_dir ./data --test_p 0.9 --test_n 2000 \
  --hidden 64 --lappe 8 --ipm_steps 16 --ipm_alpha 0.7 \
  --losstype l1 --loss_weight_x 1.0 --loss_weight_obj 3.43 --loss_weight_cons 5.8 \
  --epochs 150 --batchsize 16 --lr 4.6e-4 --seed 2026 \
  --ckpt_dir ./checkpoints
```


## License

Apache-2.0, see [LICENSE](LICENSE).
