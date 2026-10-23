"""Train an MLP on the two-moons dataset."""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from pytrainer.autodiff.losses import softmax_cross_entropy
from pytrainer.experiments.data import TwoMoonsData, iterate_minibatches, prepare_two_moons
from pytrainer.experiments.metrics import (
    HISTORY_FIELDS, evaluate, evaluate_splits, format_row, plot_curves, save_history_csv,
)
from pytrainer.nn.models import MLP
from pytrainer.nn.optim import Adam


@dataclass
class TrainConfig:
    seed: int = 42
    max_epochs: int = 300
    patience: int = 30         # stop after this many epochs without val-loss improvement; 0 = never
    min_delta: float = 1e-4    # improvement smaller than this does not reset patience
    batch_size: int = 64
    lr: float = 1e-3
    log_every: int = 10        # 0 disables progress printing


@dataclass
class TrainResult:
    model: MLP
    history: dict              # HISTORY_FIELDS -> list, one entry per epoch (0 = baseline)
    data: TwoMoonsData
    best_epoch: int            # epoch with the lowest validation loss; model holds these weights
    epochs_run: int            # epochs actually trained (< max_epochs if stopped early)
    stopped_early: bool


def make_generators(seed):
    """Independent generators for model init and batch shuffling.

    ``prepare_two_moons(seed)`` uses child 0 of ``SeedSequence(seed)`` for
    the split, so children 1 and 2 are used here.
    """
    _, init_ss, batch_ss = np.random.SeedSequence(seed).spawn(3)
    return np.random.default_rng(init_ss), np.random.default_rng(batch_ss)


def train_one_epoch(model, optimizer, X, y, batch_size, rng, after_backward=None):
    """One pass over shuffled minibatches; ``after_backward`` runs before each step."""
    for xb, yb in iterate_minibatches(X, y, batch_size, rng):
        optimizer.zero_grad()
        softmax_cross_entropy(model(xb), yb).backward()
        if after_backward is not None:
            after_backward()
        optimizer.step()  # raises on non-finite gradients


def train(config=None, data=None):
    """Train with per-epoch evaluation and validation-based model selection.

    After every epoch the model is evaluated on the full train and
    validation splits. Epoch 0 records the untrained baseline. The weights
    with the lowest validation loss are restored at the end; training stops
    early once validation loss has not improved by ``min_delta`` for
    ``patience`` epochs. The test split is never used here.
    """
    config = config or TrainConfig()
    data = data if data is not None else prepare_two_moons(seed=config.seed)

    init_rng, batch_rng = make_generators(config.seed)
    model = MLP(init_rng)
    params = model.parameters()
    optimizer = Adam(params, lr=config.lr)
    history = {k: [] for k in HISTORY_FIELDS}

    def record(epoch):
        row = {"epoch": epoch, **evaluate_splits(model, data)}
        for k in history:
            history[k].append(row[k])
        if config.log_every and (epoch <= 1 or epoch % config.log_every == 0):
            print(f"epoch {epoch:4d} | {format_row(row)}")
        return row["val_loss"]

    best_val = record(0)  # untrained baseline
    best_epoch, best_weights = 0, [p.data.copy() for p in params]
    since_best = 0
    epoch = 0

    for epoch in range(1, config.max_epochs + 1):
        train_one_epoch(model, optimizer, data.X_train, data.y_train, config.batch_size, batch_rng)
        val_loss = record(epoch)
        if val_loss < best_val - config.min_delta:
            best_val, best_epoch, best_weights = val_loss, epoch, [p.data.copy() for p in params]
            since_best = 0
        else:
            since_best += 1
        if config.patience and since_best >= config.patience:
            if config.log_every:
                print(f"early stop at epoch {epoch}")
            break

    for p, saved in zip(params, best_weights):
        p.data[...] = saved  # in place: tensor and array identity preserved
    return TrainResult(model, history, data, best_epoch, epoch, epoch < config.max_epochs)


def report(result, config, out_dir):
    """Write history.csv, both plots and summary.json; return the summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    h, best = result.history, result.best_epoch

    save_history_csv(h, os.path.join(out_dir, "history.csv"))
    for metric in ("loss", "accuracy"):
        plot_curves(h, metric, os.path.join(out_dir, f"{metric}_curve.png"), best)

    def row(i):
        return {k: h[k][i] for k in HISTORY_FIELDS[1:]}

    summary = {
        "config": asdict(config),
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "selected_epoch": best,
        "baseline": row(0),
        "last_epoch": row(-1),
        "selected": row(h["epoch"].index(best)),
        "test": evaluate(result.model, result.data.X_test, result.data.y_test),  # once, at the end
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    args = parser.parse_args(argv)

    config = TrainConfig(seed=args.seed, max_epochs=args.max_epochs,
                         patience=args.patience, lr=args.lr)
    print("config:", asdict(config))
    summary = report(train(config), config, args.out_dir)

    test = summary["test"]
    print(f"\ntrained {summary['epochs_run']} epochs"
          f"{' (early stop)' if summary['stopped_early'] else ''}; "
          f"selected epoch {summary['selected_epoch']} by validation loss")
    print(f"selected : {format_row(summary['selected'])}")
    print(f"last     : {format_row(summary['last_epoch'])}")
    print(f"test     : loss {test['loss']:.4f} acc {test['accuracy']:.4f} (n={test['n']})")
    print(f"outputs written to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
