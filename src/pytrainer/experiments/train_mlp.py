"""Train an MLP on synthetic datasets (make_moons, etc.)."""

import argparse
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

from pytrainer.autodiff.losses import softmax_cross_entropy
from pytrainer.autodiff.tensor import Tensor
from pytrainer.experiments.data import iterate_minibatches, prepare_two_moons
from pytrainer.experiments.metrics import (
    HISTORY_FIELDS,
    evaluate,
    plot_accuracy_curve,
    plot_loss_curve,
    save_history_csv,
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
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8          # PyTorch convention (see nn/optim.py)
    eval_batch_size: int = 256
    log_every: int = 10        # 0 disables progress printing


@dataclass
class TrainResult:
    model: MLP
    history: dict              # HISTORY_FIELDS -> list, one entry per epoch (0 = baseline)
    data: object               # TwoMoonsData
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


def named_parameters(model):
    names = []
    for layer_name in ("fc1", "fc2", "fc3"):
        layer = getattr(model, layer_name)
        names += [(f"{layer_name}.weight", layer.weight), (f"{layer_name}.bias", layer.bias)]
    return names


def check_finite_gradients(named_params):
    """Fail fast if any trainable parameter has a missing or non-finite grad."""
    for name, p in named_params:
        if p.grad is None:
            raise RuntimeError(f"{name} received no gradient (disconnected from the loss?)")
        if not np.all(np.isfinite(p.grad)):
            raise FloatingPointError(f"{name} has a non-finite gradient")


def train_one_epoch(model, optimizer, X, y, batch_size, rng, named_params):
    """One pass over shuffled minibatches. Returns the number of updates."""
    steps = 0
    for xb_np, yb in iterate_minibatches(X, y, batch_size, rng, shuffle=True):
        xb = Tensor(xb_np)

        optimizer.zero_grad()

        logits = model(xb)
        loss = softmax_cross_entropy(logits, yb)

        # Explicit raise, not assert: asserts vanish under `python -O`.
        if not np.all(np.isfinite(loss.data)):
            raise FloatingPointError(f"non-finite loss {loss.item()} at batch {steps}")

        loss.backward()
        check_finite_gradients(named_params)
        optimizer.step()
        steps += 1
    return steps


def _snapshot(params):
    return [p.data.copy() for p in params]


def _restore(params, snapshot):
    for p, saved in zip(params, snapshot):
        p.data[...] = saved  # in place: tensor and array identity preserved


def _record(history, epoch, train_m, val_m):
    history["epoch"].append(epoch)
    history["train_loss"].append(train_m["loss"])
    history["train_accuracy"].append(train_m["accuracy"])
    history["val_loss"].append(val_m["loss"])
    history["val_accuracy"].append(val_m["accuracy"])


def train(config=None, data=None):
    """Train with per-epoch evaluation and validation-based model selection.

    After every epoch the model is evaluated on the *full* train and
    validation splits with that epoch's final weights. Epoch 0 records the
    untrained baseline. The weights with the lowest validation loss are
    kept and restored at the end; training stops early once validation
    loss has not improved by ``min_delta`` for ``patience`` epochs.

    The test split is never used here.
    """
    config = config or TrainConfig()
    data = data if data is not None else prepare_two_moons(seed=config.seed)

    init_rng, batch_rng = make_generators(config.seed)
    model = MLP(init_rng)
    params = model.parameters()
    optimizer = Adam(params, lr=config.lr, beta1=config.beta1,
                     beta2=config.beta2, eps=config.eps)
    named_params = named_parameters(model)

    def eval_splits():
        return (
            evaluate(model, data.X_train, data.y_train, config.eval_batch_size),
            evaluate(model, data.X_val, data.y_val, config.eval_batch_size),
        )

    def log(epoch, tr, va, note=""):
        if config.log_every and (epoch <= 1 or epoch % config.log_every == 0 or note):
            print(f"epoch {epoch:4d} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
                  f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f}{note}")

    history = {k: [] for k in HISTORY_FIELDS}

    train_m, val_m = eval_splits()
    _record(history, 0, train_m, val_m)
    log(0, train_m, val_m, "  (untrained baseline)")

    best_val, best_epoch, best_weights = val_m["loss"], 0, _snapshot(params)
    since_best = 0
    epoch = 0

    for epoch in range(1, config.max_epochs + 1):
        train_one_epoch(model, optimizer, data.X_train, data.y_train,
                        config.batch_size, batch_rng, named_params)

        train_m, val_m = eval_splits()
        _record(history, epoch, train_m, val_m)

        if val_m["loss"] < best_val - config.min_delta:
            best_val, best_epoch, best_weights = val_m["loss"], epoch, _snapshot(params)
            since_best = 0
        else:
            since_best += 1

        stop = config.patience and since_best >= config.patience
        log(epoch, train_m, val_m, "  (early stop)" if stop else "")
        if stop:
            break

    _restore(params, best_weights)
    return TrainResult(
        model=model,
        history=history,
        data=data,
        best_epoch=best_epoch,
        epochs_run=epoch,
        stopped_early=epoch < config.max_epochs,
    )


def final_test_evaluation(result, batch_size=256):
    """Evaluate the selected model on the held-out test split. Call once, at the end."""
    return evaluate(result.model, result.data.X_test, result.data.y_test, batch_size)


def report(result, config, out_dir):
    """Write history.csv, both plots and summary.json; return the summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    h, b = result.history, result.best_epoch

    save_history_csv(h, os.path.join(out_dir, "history.csv"))
    plot_loss_curve(h, os.path.join(out_dir, "loss_curve.png"), best_epoch=b)
    plot_accuracy_curve(h, os.path.join(out_dir, "accuracy_curve.png"), best_epoch=b)

    test_m = final_test_evaluation(result, config.eval_batch_size)
    i = h["epoch"].index(b)
    summary = {
        "config": asdict(config),
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "selected_epoch": b,
        "baseline": {k: h[k][0] for k in HISTORY_FIELDS[1:]},
        "last_epoch": {k: h[k][-1] for k in HISTORY_FIELDS[1:]},
        "selected": {k: h[k][i] for k in HISTORY_FIELDS[1:]},
        "test": {"loss": test_m["loss"], "accuracy": test_m["accuracy"], "n": test_m["n"]},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--max-epochs", type=int, default=TrainConfig.max_epochs)
    parser.add_argument("--patience", type=int, default=TrainConfig.patience)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    args = parser.parse_args(argv)

    config = TrainConfig(seed=args.seed, max_epochs=args.max_epochs,
                         patience=args.patience, lr=args.lr)
    print("config:", asdict(config))

    result = train(config)
    summary = report(result, config, args.out_dir)

    sel, last, test = summary["selected"], summary["last_epoch"], summary["test"]
    print(f"\ntrained {summary['epochs_run']} epochs"
          f"{' (early stop)' if summary['stopped_early'] else ''}; "
          f"selected epoch {summary['selected_epoch']} by validation loss")
    print(f"selected : train loss {sel['train_loss']:.4f} acc {sel['train_accuracy']:.4f} | "
          f"val loss {sel['val_loss']:.4f} acc {sel['val_accuracy']:.4f}")
    print(f"last     : train loss {last['train_loss']:.4f} acc {last['train_accuracy']:.4f} | "
          f"val loss {last['val_loss']:.4f} acc {last['val_accuracy']:.4f}")
    print(f"test     : loss {test['loss']:.4f} acc {test['accuracy']:.4f} (n={test['n']})")
    print(f"outputs written to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()