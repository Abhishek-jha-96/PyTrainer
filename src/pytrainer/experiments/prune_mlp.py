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

Run from the directory that contains ``pytrainer/``:

    python -m pytrainer.experiments.prune_mlp --out-dir results/pruning
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from pytrainer.autodiff.losses import softmax_cross_entropy
from pytrainer.autodiff.tensor import Tensor
from pytrainer.experiments.data import iterate_minibatches, prepare_two_moons
from pytrainer.experiments.metrics import (
    evaluate, plot_accuracy_curve, plot_loss_curve, save_history_csv,
)
from pytrainer.experiments.train_mlp import (
    check_finite_gradients, make_generators, named_parameters,
)
from pytrainer.nn.models import MLP
from pytrainer.nn.optim import MaskedAdam
from pytrainer.nn.pruning import GlobalTaylorPruner, cubic_sparsity

PRUNE_FIELDS = ("epoch", "phase", "target_sparsity", "sparsity", "train_loss",
                "train_accuracy", "val_loss", "val_accuracy")

ELIGIBLE_LAYERS = ("fc1", "fc2", "fc3")
OUTPUT_WEIGHT = "fc3.weight"


@dataclass
class PruneConfig:
    seed: int = 42
    criterion: str = "taylor_abs"     # "taylor_abs" | "taylor_sq" | "magnitude"
    final_sparsity: float = 0.90      # s_f, fraction of eligible weights removed
    start_epoch: int = 10             # t0: last dense (warm-up) epoch
    end_epoch: int = 80               # t1: target sparsity reached
    finetune_epochs: int = 20         # epochs after t1 with fixed masks
    prune_every: int = 1              # epochs between mask updates in (t0, t1]
    ema_decay: float = 0.9            # saliency EMA, per minibatch
    min_keep_fraction: float = 0.05   # per-layer floor (safeguard)
    min_per_output_unit: int = 1      # every class logit keeps >= 1 input
    lr: float = 1e-3
    batch_size: int = 64
    eval_batch_size: int = 256
    log_every: int = 10

    @property
    def total_epochs(self):
        return self.end_epoch + self.finetune_epochs

    def validate(self):
        if not 0 <= self.start_epoch < self.end_epoch:
            raise ValueError("need 0 <= start_epoch < end_epoch")
        if self.finetune_epochs < 0 or self.prune_every < 1:
            raise ValueError("finetune_epochs must be >= 0 and prune_every >= 1")
        if not 0.0 <= self.final_sparsity < 1.0:
            raise ValueError("final_sparsity must be in [0, 1)")


def target_sparsity(epoch, config):
    return cubic_sparsity(epoch, config.start_epoch, config.end_epoch,
                          config.final_sparsity)


def phase_of(epoch, config):
    if epoch == 0:
        return "init"
    if epoch <= config.start_epoch:
        return "warmup"
    if epoch <= config.end_epoch:
        return "prune"
    return "finetune"


def is_prune_epoch(epoch, config):
    if not config.start_epoch < epoch <= config.end_epoch:
        return False
    return (epoch - config.start_epoch) % config.prune_every == 0 or epoch == config.end_epoch


def build_pruner(model, config):
    """Eligible weights and their layer masks, keyed "<layer>.weight".

    The mask Tensors are the layers' own ``weight_mask`` objects, so the
    pruner writes the masks the forward pass uses. Biases are not passed in.
    """
    layers = [(name, getattr(model, name)) for name in ELIGIBLE_LAYERS]
    return GlobalTaylorPruner(
        {f"{name}.weight": layer.weight for name, layer in layers},
        {f"{name}.weight": layer.weight_mask for name, layer in layers},
        ema_decay=config.ema_decay,
        criterion=config.criterion,
        min_keep_fraction=config.min_keep_fraction,
        output_name=OUTPUT_WEIGHT,
        min_per_output_unit=config.min_per_output_unit,
    )


def check_masks_hold(pruner):
    pruner.check_integrity()


def train_epoch(model, optimizer, pruner, data, config, rng, named_params, track):
    for xb_np, yb in iterate_minibatches(data.X_train, data.y_train, config.batch_size, rng):
        optimizer.zero_grad()
        loss = softmax_cross_entropy(model(Tensor(xb_np)), yb)
        if not np.all(np.isfinite(loss.data)):
            raise FloatingPointError("non-finite loss")
        loss.backward()
        check_finite_gradients(named_params)
        if track:
            pruner.observe()  # weight and gradient from the same point
        optimizer.step()


def train_with_pruning(config, data):
    """Run the full schedule. Returns a dict with model, pruner, history, selection."""
    config.validate()
    init_rng, batch_rng = make_generators(config.seed)  # same streams as dense training
    model = MLP(init_rng)
    optimizer = MaskedAdam(model.parameters(), lr=config.lr)
    named_params = named_parameters(model)
    pruner = build_pruner(model, config)

    history = {k: [] for k in PRUNE_FIELDS}
    safeguard_log = []

    def record(epoch):
        tr = evaluate(model, data.X_train, data.y_train, config.eval_batch_size)
        va = evaluate(model, data.X_val, data.y_val, config.eval_batch_size)
        target = target_sparsity(epoch, config)
        row = (epoch, phase_of(epoch, config), target, pruner.sparsity(),
               tr["loss"], tr["accuracy"], va["loss"], va["accuracy"])
        for k, v in zip(PRUNE_FIELDS, row):
            history[k].append(v)
        if config.log_every and (epoch % config.log_every == 0 or epoch == config.start_epoch):
            print(f"epoch {epoch:4d} [{row[1]:<8}] target {target:6.1%} actual {row[3]:6.1%} | "
                  f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f}")
        return va["loss"]

    record(0)
    best_val, best_epoch, best = np.inf, None, None

    for epoch in range(1, config.total_epochs + 1):
        train_epoch(model, optimizer, pruner, data, config, batch_rng, named_params,
                    track=epoch <= config.end_epoch)
        if is_prune_epoch(epoch, config):
            pruner.prune_to(target_sparsity(epoch, config), optimizer=optimizer)
            safeguard_log.append(pruner.safeguard_kept)
        check_masks_hold(pruner)
        val_loss = record(epoch)

        if epoch >= config.end_epoch and val_loss < best_val:  # select only at target sparsity
            best_val, best_epoch = val_loss, epoch
            best = [p.data.copy() for p in model.parameters()]

    if best is None:
        raise RuntimeError("no epoch at target sparsity produced a finite validation loss")
    for p, b in zip(model.parameters(), best):
        p.data[...] = b
    check_masks_hold(pruner)
    return {"model": model, "pruner": pruner, "history": history,
            "best_epoch": best_epoch, "safeguard_log": safeguard_log}


def eligible_report(pruner):
    total = sum(m.size for m in pruner.masks.values())
    kept = sum(int(m.sum()) for m in pruner.masks.values())
    return {"eligible_weights": total, "kept": kept, "pruned": total - kept,
            "sparsity": 1.0 - kept / total}


def run(config, out_dir=None):
    data = prepare_two_moons(seed=config.seed)
    res = train_with_pruning(config, data)
    model, pruner, h = res["model"], res["pruner"], res["history"]
    test = evaluate(model, data.X_test, data.y_test, config.eval_batch_size)  # once
    i = h["epoch"].index(res["best_epoch"])

    summary = {
        "config": asdict(config),
        "total_epochs": config.total_epochs,
        "selected_epoch": res["best_epoch"],
        "budget": eligible_report(pruner),
        "biases_dense": int(sum(p.data.size for n, p in named_parameters(model)
                                if n.endswith(".bias"))),
        "safeguard_kept_max": max(res["safeguard_log"], default=0),
        "selected": {k: h[k][i] for k in PRUNE_FIELDS[2:]},
        "test": test,
        "layers": pruner.layer_report(),
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_history_csv(h, os.path.join(out_dir, "pruning_history.csv"), fields=PRUNE_FIELDS)
        plot_loss_curve(h, os.path.join(out_dir, "pruning_loss_curve.png"), res["best_epoch"])
        plot_accuracy_curve(h, os.path.join(out_dir, "pruning_accuracy_curve.png"), res["best_epoch"])
        plot_schedule(h, config, os.path.join(out_dir, "sparsity_schedule.png"))
        with open(os.path.join(out_dir, "pruning_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=float)
    return summary, res


def plot_schedule(history, config, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.linspace(0, config.total_epochs, 500)
    s = [target_sparsity(x, config) for x in t]

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


def main(argv=None):
    p = argparse.ArgumentParser(description="Gradual global saliency pruning")
    p.add_argument("--out-dir", default="results/pruning")
    p.add_argument("--criterion", default=PruneConfig.criterion,
                   choices=("taylor_abs", "taylor_sq", "magnitude"))
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