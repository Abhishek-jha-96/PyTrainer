"""Gradual global saliency pruning of the trained two-moons MLP.

Pipeline:
    1. Dense training (train_mlp.train), restored to its best-val epoch.
    2. Gradual pruning: train with MaskedAdam, update the saliency EMA every
       minibatch, and raise sparsity on a cubic schedule every
       ``prune_every`` epochs until ``final_sparsity``.
    3. Fine-tuning with masks fixed; keep the best-val-loss epoch.
    4. One evaluation of the selected sparse model on the test split.

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
    TrainConfig, check_finite_gradients, named_parameters, train,
)
from pytrainer.nn.optim import MaskedAdam
from pytrainer.nn.pruning import Pruner, cubic_sparsity

PRUNE_FIELDS = ("epoch", "phase", "sparsity", "train_loss", "train_accuracy",
                "val_loss", "val_accuracy")


@dataclass
class PruneConfig:
    seed: int = 42
    criterion: str = "taylor_abs"   # "taylor_abs" | "taylor_sq" | "magnitude"
    final_sparsity: float = 0.90    # fraction of *weights* removed (biases stay dense)
    num_prune_events: int = 15      # cubic schedule steps
    prune_every: int = 2            # epochs between pruning events
    finetune_epochs: int = 40       # masks fixed; best val loss is kept
    ema_decay: float = 0.9          # per minibatch (~22 batches per epoch)
    min_keep_fraction: float = 0.05 # per-layer floor
    min_per_output_unit: int = 1    # every class logit keeps >= 1 input
    lr: float = 1e-3
    batch_size: int = 64
    eval_batch_size: int = 256
    log_every: int = 5


def weight_names(model):
    return [(n, p) for n, p in named_parameters(model) if n.endswith(".weight")]


def train_epoch_with_saliency(model, optimizer, pruner, data, batch_size, rng, named_params):
    for xb_np, yb in iterate_minibatches(data.X_train, data.y_train, batch_size, rng):
        optimizer.zero_grad()
        loss = softmax_cross_entropy(model(Tensor(xb_np)), yb)
        if not np.all(np.isfinite(loss.data)):
            raise FloatingPointError("non-finite loss during pruning")
        loss.backward()
        check_finite_gradients(named_params)
        pruner.update_scores()   # weight and gradient from the same point
        optimizer.step()


def check_masks_hold(pruner):
    for n, w in pruner.named_weights:
        if np.any(w.data[~pruner.masks[n]] != 0.0):
            raise AssertionError(f"pruned weights of {n} are no longer zero")


def prune_model(dense, data, config):
    """Run phases 2 and 3 on ``dense`` (modified in place). Returns a result dict."""
    model = dense
    # SeedSequence children: 0 split, 1 init, 2 dense batches, 3 pruning batches.
    rng = np.random.default_rng(np.random.SeedSequence(config.seed).spawn(4)[3])

    optimizer = MaskedAdam(model.parameters(), lr=config.lr)
    named = weight_names(model)
    pruner = Pruner(named, optimizer, criterion=config.criterion, decay=config.ema_decay,
                    min_keep_fraction=config.min_keep_fraction,
                    output_layer=named[-1][0],
                    min_per_output_unit=config.min_per_output_unit)
    named_params = named_parameters(model)
    history = {k: [] for k in PRUNE_FIELDS}

    def record(epoch, phase):
        tr = evaluate(model, data.X_train, data.y_train, config.eval_batch_size)
        va = evaluate(model, data.X_val, data.y_val, config.eval_batch_size)
        s = pruner.sparsity()
        for k, v in zip(PRUNE_FIELDS, (epoch, phase, s, tr["loss"], tr["accuracy"],
                                       va["loss"], va["accuracy"])):
            history[k].append(v)
        if config.log_every and (epoch % config.log_every == 0 or phase == "dense"):
            print(f"epoch {epoch:4d} [{phase:<8}] sparsity {s:6.1%} | "
                  f"train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f}")
        return va["loss"]

    record(0, "dense")

    # Phase 2: gradual pruning
    epoch = 0
    for event in range(1, config.num_prune_events + 1):
        for _ in range(config.prune_every):
            epoch += 1
            train_epoch_with_saliency(model, optimizer, pruner, data,
                                      config.batch_size, rng, named_params)
            if _ < config.prune_every - 1:
                record(epoch, "prune")
        target = cubic_sparsity(event, config.num_prune_events, 0.0, config.final_sparsity)
        pruner.prune_to(target)
        check_masks_hold(pruner)
        record(epoch, "prune")

    # Phase 3: fine-tune with fixed masks, keep best val loss
    best_val, best_epoch = np.inf, epoch
    best = [p.data.copy() for p in model.parameters()]
    for _ in range(config.finetune_epochs):
        epoch += 1
        train_epoch_with_saliency(model, optimizer, pruner, data,
                                  config.batch_size, rng, named_params)
        check_masks_hold(pruner)
        v = record(epoch, "finetune")
        if v < best_val:
            best_val, best_epoch = v, epoch
            best = [p.data.copy() for p in model.parameters()]
    for p, b in zip(model.parameters(), best):
        p.data[...] = b
    check_masks_hold(pruner)

    return {"model": model, "pruner": pruner, "history": history, "best_epoch": best_epoch}


def run(config, out_dir=None):
    data = prepare_two_moons(seed=config.seed)
    dense_result = train(TrainConfig(seed=config.seed, log_every=0), data=data)
    dense = dense_result.model
    dense_val = evaluate(dense, data.X_val, data.y_val)
    dense_test = evaluate(dense, data.X_test, data.y_test)  # reported in Subtask 7 already
    print(f"dense model: epoch {dense_result.best_epoch}, "
          f"val acc {dense_val['accuracy']:.4f}, val loss {dense_val['loss']:.4f}")

    res = prune_model(dense, data, config)
    model, pruner, h = res["model"], res["pruner"], res["history"]
    test = evaluate(model, data.X_test, data.y_test)  # single final evaluation
    i = h["epoch"].index(res["best_epoch"], len(h["epoch"]) - config.finetune_epochs)

    summary = {
        "config": asdict(config),
        "dense": {"epoch": dense_result.best_epoch, "val": dense_val, "test": dense_test,
                  "weights": int(sum(m.size for m in pruner.masks.values()))},
        "pruned": {
            "selected_epoch": res["best_epoch"],
            "sparsity": pruner.sparsity(),
            "weights_kept": int(sum(m.sum() for m in pruner.masks.values())),
            "val": {"loss": h["val_loss"][i], "accuracy": h["val_accuracy"][i]},
            "train": {"loss": h["train_loss"][i], "accuracy": h["train_accuracy"][i]},
            "test": test,
            "layers": pruner.layer_report(),
        },
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        save_history_csv(h, os.path.join(out_dir, "pruning_history.csv"), fields=PRUNE_FIELDS)
        plot_loss_curve(h, os.path.join(out_dir, "pruning_loss_curve.png"), res["best_epoch"])
        plot_accuracy_curve(h, os.path.join(out_dir, "pruning_accuracy_curve.png"), res["best_epoch"])
        _plot_sparsity(h, os.path.join(out_dir, "sparsity_schedule.png"))
        with open(os.path.join(out_dir, "pruning_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2, default=float)

    return summary, res


def _plot_sparsity(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history["sparsity"], drawstyle="steps-post", label="weight sparsity")
    ax.set_xlabel("epoch (after dense training)")
    ax.set_ylabel("fraction of weights pruned")
    ax2 = ax.twinx()
    ax2.plot(history["epoch"], history["val_accuracy"], color="tab:orange", label="val accuracy")
    ax2.set_ylabel("validation accuracy")
    ax.set_title("Sparsity schedule and validation accuracy")
    ax.grid(alpha=0.3)
    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.15))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Gradual global saliency pruning")
    p.add_argument("--out-dir", default="results/pruning")
    p.add_argument("--criterion", default=PruneConfig.criterion,
                   choices=("taylor_abs", "taylor_sq", "magnitude"))
    p.add_argument("--final-sparsity", type=float, default=PruneConfig.final_sparsity)
    p.add_argument("--seed", type=int, default=PruneConfig.seed)
    a = p.parse_args(argv)
    config = PruneConfig(seed=a.seed, criterion=a.criterion, final_sparsity=a.final_sparsity)
    print("config:", asdict(config))
    summary, _ = run(config, a.out_dir)
    pr = summary["pruned"]
    print(f"\nselected epoch {pr['selected_epoch']}: sparsity {pr['sparsity']:.1%} "
          f"({pr['weights_kept']}/{summary['dense']['weights']} weights kept)")
    print(f"val  acc {pr['val']['accuracy']:.4f} loss {pr['val']['loss']:.4f}")
    print(f"test acc {pr['test']['accuracy']:.4f} loss {pr['test']['loss']:.4f} "
          f"(dense test acc {summary['dense']['test']['accuracy']:.4f})")
    for n, info in pr["layers"].items():
        print(f"  {n}: kept {info['kept']}/{info['total']}, "
              f"units w/o inputs {info['units_without_inputs']}"
              + (f", w/o outputs {info['units_without_outputs']}"
                 if "units_without_outputs" in info else ""))


if __name__ == "__main__":
    main()