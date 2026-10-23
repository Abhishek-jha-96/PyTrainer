"""Train the two-moons MLP with gradual global saliency pruning.

One run from initialization, epochs 1 .. t1 + finetune_epochs:

    warm-up     t <= t0        dense training; saliency EMA warms up
    pruning     t0 < t <= t1   after each epoch, prune to s(t)
    fine-tune   t > t1         masks fixed at s_f

    s(t) = s_f * [1 - (1 - u)^3],   u = clip((t - t0) / (t1 - t0), 0, 1)

Only weights are eligible (608 in the 2-32-16-2 MLP); biases are excluded
from the pruning budget. Ranking is global across fc1, fc2 and fc3, and each
prune removes an exact number of connections, floor(s * 608 + 0.5).
The model is selected by validation loss among epochs with t >= t1
(target sparsity reached); the test split is evaluated once at the end.

    python -m pytrainer.experiments.prune_mlp --out-dir results/pruning
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np

from pytrainer.experiments.data import prepare_two_moons
from pytrainer.experiments.metrics import (
    evaluate, evaluate_splits, format_row, plot_curves, save_history_csv,
)
from pytrainer.experiments.train_mlp import make_generators, train_one_epoch
from pytrainer.nn.models import MLP
from pytrainer.nn.optim import MaskedAdam
from pytrainer.nn.pruning import CRITERIA, GlobalTaylorPruner, cubic_sparsity


@dataclass
class PruneConfig:
    seed: int = 42
    criterion: str = "taylor_abs"     # one of pruning.CRITERIA
    final_sparsity: float = 0.90      # s_f, fraction of eligible weights removed
    start_epoch: int = 10             # t0: last dense (warm-up) epoch
    end_epoch: int = 80               # t1: target sparsity reached
    finetune_epochs: int = 20         # epochs after t1 with fixed masks
    ema_decay: float = 0.9            # saliency EMA, per minibatch
    min_keep_fraction: float = 0.05   # per-layer floor (safeguard)
    min_per_output_unit: int = 1      # every class logit keeps >= 1 input
    lr: float = 1e-3
    batch_size: int = 64
    log_every: int = 10

    @property
    def total_epochs(self):
        return self.end_epoch + self.finetune_epochs

    def target_sparsity(self, epoch):
        return cubic_sparsity(epoch, self.start_epoch, self.end_epoch, self.final_sparsity)

    def phase(self, epoch):
        if epoch == 0:
            return "init"
        if epoch <= self.start_epoch:
            return "warmup"
        if epoch <= self.end_epoch:
            return "prune"
        return "finetune"


def build_pruner(model, config):
    """Pruner over each layer's weight and its own ``weight_mask`` Tensor,
    so the pruner writes the masks the forward pass uses. Biases are excluded."""
    layers = [(name, getattr(model, name)) for name in MLP.LAYERS]
    return GlobalTaylorPruner(
        {f"{name}.weight": layer.weight for name, layer in layers},
        {f"{name}.weight": layer.weight_mask for name, layer in layers},
        ema_decay=config.ema_decay,
        criterion=config.criterion,
        min_keep_fraction=config.min_keep_fraction,
        output_name="fc3.weight",
        min_per_output_unit=config.min_per_output_unit,
    )


def train_with_pruning(config, data):
    """Run the full schedule. Returns a dict with model, pruner, history, selection."""
    init_rng, batch_rng = make_generators(config.seed)  # same streams as dense training
    model = MLP(init_rng)
    params = model.parameters()
    optimizer = MaskedAdam(params, lr=config.lr)
    pruner = build_pruner(model, config)
    history = {k: [] for k in ("epoch", "phase", "target_sparsity", "sparsity", "train_loss",
                               "train_accuracy", "val_loss", "val_accuracy")}

    def record(epoch):
        row = {"epoch": epoch, "phase": config.phase(epoch),
               "target_sparsity": config.target_sparsity(epoch),
               "sparsity": pruner.sparsity(), **evaluate_splits(model, data)}
        for k in history:
            history[k].append(row[k])
        if config.log_every and (epoch % config.log_every == 0 or epoch == config.start_epoch):
            print(f"epoch {epoch:4d} [{row['phase']:<8}] target {row['target_sparsity']:6.1%} "
                  f"actual {row['sparsity']:6.1%} | {format_row(row)}")
        return row["val_loss"]

    record(0)
    best_val, best_epoch, best_weights = np.inf, None, None
    safeguard_kept_max = 0

    for epoch in range(1, config.total_epochs + 1):
        pruning = epoch <= config.end_epoch
        train_one_epoch(model, optimizer, data.X_train, data.y_train, config.batch_size,
                        batch_rng, after_backward=pruner.observe if pruning else None)
        if epoch > config.start_epoch and pruning:
            pruner.prune_to(config.target_sparsity(epoch), optimizer=optimizer)
            safeguard_kept_max = max(safeguard_kept_max, pruner.safeguard_kept)
        pruner.check_integrity(optimizer)
        val_loss = record(epoch)

        if epoch >= config.end_epoch and val_loss < best_val:  # select only at target sparsity
            best_val, best_epoch = val_loss, epoch
            best_weights = [p.data.copy() for p in params]

    for p, saved in zip(params, best_weights):
        p.data[...] = saved
    return {"model": model, "pruner": pruner, "history": history,
            "best_epoch": best_epoch, "safeguard_kept_max": safeguard_kept_max}


def plot_schedule(history, config, path):
    t = np.linspace(0, config.total_epochs, 500)
    s = [config.target_sparsity(x) for x in t]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 2]})
    for a in (ax, ax2):
        a.axvspan(0, config.start_epoch, color="tab:green", alpha=0.08)
        a.axvspan(config.end_epoch, config.total_epochs, color="tab:purple", alpha=0.08)
        a.grid(alpha=0.3)
    ax.plot(t, s, color="gray", linestyle="--", label="schedule s(t)")
    ax.plot(history["epoch"], history["sparsity"], drawstyle="steps-post",
            label="measured sparsity")
    ax.set_ylabel("fraction of eligible weights pruned")
    ax.set_title(f"Cubic schedule: warm-up to t0={config.start_epoch}, "
                 f"s_f={config.final_sparsity:.0%} at t1={config.end_epoch}")
    ax.text(config.start_epoch / 2, 0.45, "warm-up", ha="center", color="tab:green")
    ax.text((config.end_epoch + config.total_epochs) / 2, 0.45, "fine-tune",
            ha="center", color="tab:purple")
    ax.legend(loc="lower right")
    ax2.plot(history["epoch"], history["train_accuracy"], label="train")
    ax2.plot(history["epoch"], history["val_accuracy"], label="validation")
    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("epoch")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(config, out_dir=None):
    data = prepare_two_moons(seed=config.seed)
    res = train_with_pruning(config, data)
    model, pruner, h, best = res["model"], res["pruner"], res["history"], res["best_epoch"]
    layers = pruner.layer_report()
    total = sum(r["total"] for r in layers.values())
    kept = sum(r["kept"] for r in layers.values())

    summary = {
        "config": asdict(config),
        "total_epochs": config.total_epochs,
        "selected_epoch": best,
        "budget": {"eligible_weights": total, "kept": kept, "pruned": total - kept,
                   "sparsity": 1.0 - kept / total},
        "biases_dense": sum(p.data.size for n, p in model.named_parameters()
                            if n.endswith(".bias")),
        "safeguard_kept_max": res["safeguard_kept_max"],
        "selected": {k: v[h["epoch"].index(best)] for k, v in h.items() if k not in ("epoch", "phase")},
        "test": evaluate(model, data.X_test, data.y_test),  # once, at the end
        "layers": layers,
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_history_csv(h, os.path.join(out_dir, "pruning_history.csv"))
        for metric in ("loss", "accuracy"):
            plot_curves(h, metric, os.path.join(out_dir, f"pruning_{metric}_curve.png"), best)
        plot_schedule(h, config, os.path.join(out_dir, "sparsity_schedule.png"))
        with open(os.path.join(out_dir, "pruning_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
    return summary, res


def main(argv=None):
    p = argparse.ArgumentParser(description="Gradual global saliency pruning")
    p.add_argument("--out-dir", default="results/pruning")
    p.add_argument("--criterion", default=PruneConfig.criterion, choices=CRITERIA)
    p.add_argument("--final-sparsity", type=float, default=PruneConfig.final_sparsity)
    p.add_argument("--start-epoch", type=int, default=PruneConfig.start_epoch)
    p.add_argument("--end-epoch", type=int, default=PruneConfig.end_epoch)
    p.add_argument("--finetune-epochs", type=int, default=PruneConfig.finetune_epochs)
    p.add_argument("--seed", type=int, default=PruneConfig.seed)
    a = p.parse_args(argv)
    config = PruneConfig(seed=a.seed, criterion=a.criterion, final_sparsity=a.final_sparsity,
                         start_epoch=a.start_epoch, end_epoch=a.end_epoch,
                         finetune_epochs=a.finetune_epochs)
    print("config:", asdict(config))
    s, _ = run(config, a.out_dir)
    b, sel = s["budget"], s["selected"]
    print(f"\n{s['total_epochs']} epochs; selected epoch {s['selected_epoch']}")
    print(f"budget: {b['pruned']}/{b['eligible_weights']} eligible weights pruned "
          f"({b['sparsity']:.2%}), {b['kept']} kept; {s['biases_dense']} biases dense")
    print(f"val  acc {sel['val_accuracy']:.4f} loss {sel['val_loss']:.4f}")
    print(f"test acc {s['test']['accuracy']:.4f} loss {s['test']['loss']:.4f}")
    print(f"safeguard-only survivors (max over prunes): {s['safeguard_kept_max']}")
    for n, info in s["layers"].items():
        print(f"  {n}: kept {info['kept']}/{info['total']} ({1 - info['sparsity']:.1%})")


if __name__ == "__main__":
    main()
