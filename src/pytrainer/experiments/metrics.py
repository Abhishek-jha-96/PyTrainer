"""Evaluation metrics computed with fixed parameters (no backward, no step)."""

import numpy as np

from pytrainer.autodiff.losses import softmax_cross_entropy
from pytrainer.autodiff.tensor import Tensor
from pytrainer.experiments.data import iterate_minibatches


def accuracy(logits, labels):
    """Fraction of rows whose argmax matches the integer label."""
    logits = logits.data if isinstance(logits, Tensor) else np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim != 2 or labels.shape != (logits.shape[0],):
        raise ValueError(
            f"expected logits (N, C) and labels (N,); got {logits.shape} and {labels.shape}"
        )
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == labels))


def evaluate(model, X, y, batch_size=256):
    """Dataset-level mean loss and accuracy under the model's current weights.

    The loss is the per-sample mean over the whole dataset: each batch's
    mean loss is weighted by its sample count, so the result does not
    depend on ``batch_size`` (including a smaller final batch).

    Never calls ``backward()`` or ``optimizer.step()``. The engine still
    builds a graph for each batch's forward pass; it is dropped when the
    batch goes out of scope. Parameter values and ``.grad`` are untouched.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n = len(X)
    if n == 0:
        raise ValueError("cannot evaluate on an empty dataset")

    loss_sum = 0.0
    correct = 0
    for xb, yb in iterate_minibatches(X, y, batch_size, shuffle=False):
        logits = model(Tensor(xb))
        batch_loss = softmax_cross_entropy(logits, yb).item()  # mean over batch
        loss_sum += batch_loss * len(xb)
        correct += int(np.sum(np.argmax(logits.data, axis=1) == yb))

    return {"loss": loss_sum / n, "accuracy": correct / n, "n": n}


# ----------------------------------------------------------------------
# Learning-curve reporting
# ----------------------------------------------------------------------
HISTORY_FIELDS = ("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy")


def save_history_csv(history, path, fields=HISTORY_FIELDS):
    """Write one row per epoch with the given columns (first column = integer epoch)."""
    import csv
 
    lengths = {len(history[k]) for k in fields}
    if len(lengths) != 1:
        raise ValueError(f"history columns have different lengths: {lengths}")
 
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for row in zip(*(history[k] for k in fields)):
            epoch, *values = row
            writer.writerow([int(epoch)] + [v if isinstance(v, str) else repr(float(v)) for v in values])


def load_history_csv(path):
    """Read a CSV written by ``save_history_csv`` back into a history dict."""
    import csv

    history = {k: [] for k in HISTORY_FIELDS}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames) != HISTORY_FIELDS:
            raise ValueError(f"unexpected CSV columns: {reader.fieldnames}")
        for row in reader:
            history["epoch"].append(int(row["epoch"]))
            for k in HISTORY_FIELDS[1:]:
                history[k].append(float(row[k]))
    return history


def _plot_curves(history, train_key, val_key, ylabel, title, path, best_epoch=None):
    import matplotlib

    matplotlib.use("Agg")  # file output only; no display needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history[train_key], label="train", linewidth=1.8)
    ax.plot(history["epoch"], history[val_key], label="validation", linewidth=1.8)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                   label=f"selected epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss_curve(history, path, best_epoch=None):
    _plot_curves(history, "train_loss", "val_loss", "mean cross-entropy",
                 "Loss curve (full-split evaluation after each epoch)", path, best_epoch)


def plot_accuracy_curve(history, path, best_epoch=None):
    _plot_curves(history, "train_accuracy", "val_accuracy", "accuracy",
                 "Accuracy curve (full-split evaluation after each epoch)", path, best_epoch)