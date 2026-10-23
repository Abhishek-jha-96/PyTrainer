# PyTrainer

A NumPy-only autodiff engine, an MLP trained with it on two moons, gradual
global Taylor-saliency pruning, and the evidence that the pruning pays off.

| part | what | code |
|---|---|---|
| 1 | Autodiff engine (`Tensor`, ops, softmax cross-entropy) | `src/pytrainer/autodiff/` |
| 2 | 2-32-16-2 MLP trained with Adam, early stopping on validation loss | `experiments/train_mlp.py` |
| 3 | Self-pruning: gradual global saliency pruning to 90% sparsity | `nn/pruning.py`, `experiments/prune_mlp.py` |
| 4 | Pareto sweep: Taylor vs magnitude across sparsities and seeds, real cost | `experiments/sweep.py`, `nn/sparse.py` |

Design rationale (criterion derivation, masked gradients, engine bottlenecks,
serving at scale) is in [`DESIGN.md`](DESIGN.md).

## Installation

Requires Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd PyTrainer
uv sync
```

`uv sync` creates `.venv` and installs the pinned dependencies from `uv.lock`
(numpy, scipy, scikit-learn, matplotlib, pytest). Every command below runs from
the repository root.

## Gradient-check tests

Compares every op's analytic gradient against central finite differences:

```bash
uv run pytest src/pytrainer/tests/test_gradient_check.py
```

## Part 2: training

```bash
uv run python -m pytrainer.experiments.train_mlp --out-dir results
```

Trains with seed 42 (up to 300 epochs, early stopping with patience 30) and
writes `history.csv`, `loss_curve.png`, `accuracy_curve.png` and `summary.json`
to `results/`. The test split is evaluated once, at the end.

## Part 3: self-pruning run

```bash
uv run python -m pytrainer.experiments.prune_mlp --out-dir results/pruning
```

The run has three phases: 10 dense warm-up epochs, then pruning after every epoch along a cubic schedule to 90% sparsity at epoch 80, then 20 fine-tuning epochs. Weights are ranked globally by |w·g|, averaged over minibatches. Outputs go to `results/pruning/`:
- `pruning_history.csv`
- the loss and accuracy curves
- `sparsity_schedule.png`
- `pruning_summary.json`

Use `--criterion magnitude` for magnitude pruning or `--final-sparsity 0.95` for a different target; `--help` lists all options.

## Part 4: Pareto sweep

```bash
uv run python -m pytrainer.experiments.sweep --out-dir results/sweep
```

Trains 270 models: 0/50/75/90/95% sparsity × {Taylor, magnitude} × 30 seeds. It takes about 4 minutes on 8 cores. Outputs go to `results/sweep/`:
- `runs.csv`, `summary.csv`, `comparison.csv`
- `cost.csv`: wall-clock comparison of dense, sparse and compacted forward passes
- `pareto.png`

The analysis and the falsifiable claim are in [`results/sweep/README.md`](results/sweep/README.md). The held-out replication used seeds 30–59:

```bash
uv run python -m pytrainer.experiments.sweep --first-seed 30 --out-dir results/sweep_replication
```

Results are deterministic for a given seed. The only exception is the wall-clock timings in `cost.csv`, which vary by machine.
