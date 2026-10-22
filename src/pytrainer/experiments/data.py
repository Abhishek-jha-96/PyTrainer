from dataclasses import dataclass

import numpy as np

SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


# ----------------------------------------------------------------------
# 1. Generation
# ----------------------------------------------------------------------
def make_moons_numpy(n_samples, noise, rng):
    """NumPy-only two moons, following sklearn's construction.

    Outer moon (label 0): (cos t, sin t),           t in [0, pi]
    Inner moon (label 1): (1 - cos t, 1 - sin t - 0.5)
    Gaussian noise with std ``noise`` is added to both coordinates.
    """
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0.0, np.pi, n_out)
    t_in = np.linspace(0.0, np.pi, n_in)

    X = np.vstack([
        np.column_stack([np.cos(t_out), np.sin(t_out)]),
        np.column_stack([1.0 - np.cos(t_in), 1.0 - np.sin(t_in) - 0.5]),
    ])
    y = np.concatenate([np.zeros(n_out), np.ones(n_in)]).astype(np.int64)

    if noise:
        X = X + rng.normal(0.0, noise, size=X.shape)

    order = rng.permutation(n_samples)
    return X[order], y[order]


def generate_two_moons(n_samples=2000, noise=0.2, random_state=42, backend="sklearn"):
    """Generate raw two-moons data.

    Returns:
        X: float64 array, shape (n_samples, 2)
        y: int64 array of class indices {0, 1}, shape (n_samples,)
    """
    if backend == "sklearn":
        try:
            from sklearn.datasets import make_moons
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is not installed; use backend='numpy'"
            ) from exc
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif backend == "numpy":
        X, y = make_moons_numpy(n_samples, noise, np.random.default_rng(random_state))
    else:
        raise ValueError(f"unknown backend {backend!r}; use 'sklearn' or 'numpy'")

    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


# ----------------------------------------------------------------------
# 2. Stratified split
# ----------------------------------------------------------------------
def stratified_split(y, rng, fractions=SPLIT_FRACTIONS):
    """Split sample indices into train/val/test, preserving class ratios.

    Returns:
        dict with "train", "val", "test" -> sorted-then-shuffled int index arrays.
    """
    fractions = tuple(fractions)
    if len(fractions) != 3 or any(f <= 0 for f in fractions):
        raise ValueError(f"need three positive fractions, got {fractions}")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")

    y = np.asarray(y)
    parts = {"train": [], "val": [], "test": []}

    for cls in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == cls))
        n_train = int(round(fractions[0] * len(idx)))
        n_val = int(round(fractions[1] * len(idx)))
        parts["train"].append(idx[:n_train])
        parts["val"].append(idx[n_train:n_train + n_val])
        parts["test"].append(idx[n_train + n_val:])

    # Interleave classes within each split.
    return {name: rng.permutation(np.concatenate(chunks)) for name, chunks in parts.items()}


# ----------------------------------------------------------------------
# 3. Standardization (fit on train only)
# ----------------------------------------------------------------------
class Standardizer:
    """Per-feature z-scoring with statistics fitted on the training split."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=np.float64)
        if X_train.ndim != 2 or len(X_train) == 0:
            raise ValueError(f"expected non-empty (N, D) array, got {X_train.shape}")
        self.mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        # Guard constant features: leave them centred but unscaled.
        self.std = np.where(std > 0.0, std, 1.0)
        return self

    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("Standardizer.transform called before fit")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"expected shape (N, {self.mean.shape[0]}), got {X.shape}"
            )
        return (X - self.mean) / self.std


# ----------------------------------------------------------------------
# 4. Prepared dataset
# ----------------------------------------------------------------------
@dataclass
class TwoMoonsData:
    X_train: np.ndarray  # (n_train, 2) float64, standardized
    y_train: np.ndarray  # (n_train,)   int64 in {0, 1}
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Standardizer
    indices: dict        # split name -> indices into the raw arrays
    X_raw: np.ndarray    # unscaled inputs, kept for plotting
    y_raw: np.ndarray


def prepare_two_moons(
    seed=42,
    n_samples=2000,
    noise=0.2,
    backend="sklearn",
    fractions=SPLIT_FRACTIONS,
):
    """Generate, split and standardize two moons, reproducibly from ``seed``.

    """
    X, y = generate_two_moons(n_samples, noise, random_state=seed, backend=backend)

    split_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    idx = stratified_split(y, split_rng, fractions)

    scaler = Standardizer().fit(X[idx["train"]])

    data = TwoMoonsData(
        X_train=scaler.transform(X[idx["train"]]),
        y_train=y[idx["train"]],
        X_val=scaler.transform(X[idx["val"]]),
        y_val=y[idx["val"]],
        X_test=scaler.transform(X[idx["test"]]),
        y_test=y[idx["test"]],
        scaler=scaler,
        indices=idx,
        X_raw=X,
        y_raw=y,
    )

    for name in ("X_train", "X_val", "X_test"):
        if not np.all(np.isfinite(getattr(data, name))):
            raise FloatingPointError(f"{name} contains non-finite values")

    return data


# ----------------------------------------------------------------------
# 5. Minibatches
# ----------------------------------------------------------------------
def iterate_minibatches(X, y, batch_size=64, rng=None, shuffle=True):
    """Yield ``(X_batch, y_batch)`` slices, including the final partial batch.

    """
    X = np.asarray(X)
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"X and y lengths differ: {len(X)} vs {len(y)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    n = len(X)
    if shuffle:
        if rng is None:
            raise ValueError("shuffle=True requires a seeded rng")
        order = rng.permutation(n)
    else:
        order = np.arange(n)

    for start in range(0, n, batch_size):
        batch = order[start:start + batch_size]
        yield X[batch], y[batch]