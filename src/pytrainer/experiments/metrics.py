"""Evaluation metrics and learning-curve reporting."""

import csv

import matplotlib.pyplot as plt
import numpy as np

from pytrainer.autodiff.losses import softmax_cross_entropy

HISTORY_FIELDS = ("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy")


def accuracy(logits, labels):
    """Fraction of rows whose argmax matches the integer label."""
    return float(np.mean(np.argmax(logits.data, axis=1) == labels))


def evaluate(model, X, y):
    """Mean loss and accuracy over the whole dataset under the current weights.

    One full-batch forward pass; never calls ``backward()`` or ``step()``,
    so parameter values and ``.grad`` are untouched.
    """
    logits = model(X)
    return {
        "loss": softmax_cross_entropy(logits, y).item(),
        "accuracy": accuracy(logits, y),
        "n": len(y),
    }


def evaluate_splits(model, data):
    """Train and validation metrics as a flat history row."""
    tr = evaluate(model, data.X_train, data.y_train)
    va = evaluate(model, data.X_val, data.y_val)
    return {"train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
            "val_loss": va["loss"], "val_accuracy": va["accuracy"]}


def format_row(row):
    return (f"train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} | "
            f"val loss {row['val_loss']:.4f} acc {row['val_accuracy']:.4f}")


def save_history_csv(history, path):
    """Write one row per epoch, columns in ``history`` order."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(history)
        writer.writerows(zip(*history.values()))


def plot_curves(history, metric, path, best_epoch):
    """Train vs validation ``metric`` ("loss" or "accuracy") per epoch."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["epoch"], history[f"train_{metric}"], label="train", linewidth=1.8)
    ax.plot(history["epoch"], history[f"val_{metric}"], label="validation", linewidth=1.8)
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
               label=f"selected epoch ({best_epoch})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean cross-entropy" if metric == "loss" else metric)
    ax.set_title(f"{metric.capitalize()} curve (full-split evaluation after each epoch)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
