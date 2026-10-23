from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_moons

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def stratified_split(y, rng):
    """Split sample indices into shuffled train/val/test, preserving class ratios."""
    parts = {"train": [], "val": [], "test": []}
    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n_train = int(round(SPLIT_FRACTIONS[0] * len(idx)))
        n_val = int(round(SPLIT_FRACTIONS[1] * len(idx)))
        parts["train"].append(idx[:n_train])
        parts["val"].append(idx[n_train:n_train + n_val])
        parts["test"].append(idx[n_train + n_val:])
    # Interleave classes within each split.
    return {name: rng.permutation(np.concatenate(chunks)) for name, chunks in parts.items()}


@dataclass
class TwoMoonsData:
    X_train: np.ndarray  # (n_train, 2) float64, standardized
    y_train: np.ndarray  # (n_train,)   int64 in {0, 1}
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray     # train-split statistics, to standardize new samples
    std: np.ndarray


def prepare_two_moons(seed=42, n_samples=2000, noise=0.2):
    """Generate, split and standardize two moons, reproducibly from ``seed``.

    Standardization statistics come from the training split only.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    y = y.astype(np.int64)

    # Child 0 of SeedSequence(seed); training uses children 1 and 2.
    split_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    idx = stratified_split(y, split_rng)

    mean, std = X[idx["train"]].mean(axis=0), X[idx["train"]].std(axis=0)
    split = {name: ((X[i] - mean) / std, y[i]) for name, i in idx.items()}
    return TwoMoonsData(*split["train"], *split["val"], *split["test"], mean, std)


def iterate_minibatches(X, y, batch_size, rng):
    """Yield shuffled ``(X_batch, y_batch)`` slices, including the final partial batch."""
    order = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch = order[start:start + batch_size]
        yield X[batch], y[batch]
