import numpy as np

from pytrainer.autodiff.ops import _node
from pytrainer.autodiff.tensor import Tensor


def softmax_cross_entropy(logits, targets):
    """Numerically stable mean softmax cross-entropy.

    logits:  Tensor of raw class scores, shape (N, C).
    targets: integer class indices, shape (N,), values in [0, C).

    Returns a scalar Tensor. Only logits receive gradients:
    dL/dlogits = (softmax(logits) - onehot(targets)) / N.
    """
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a Tensor")
    if logits.ndim != 2 or 0 in logits.shape:
        raise ValueError(f"logits must have non-empty shape (N, C); got {logits.shape}")

    batch_size, num_classes = logits.shape
    targets = np.asarray(targets)
    if targets.shape != (batch_size,):
        raise ValueError(f"targets must have shape ({batch_size},); got {targets.shape}")
    if not np.issubdtype(targets.dtype, np.integer):
        raise TypeError("targets must contain integer class indices")
    if np.any((targets < 0) | (targets >= num_classes)):
        raise ValueError("target class indices are out of range")

    # Stable log-softmax via log-sum-exp.
    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    rows = np.arange(batch_size)
    loss = -log_probs[rows, targets].mean()

    grad = np.exp(log_probs)
    grad[rows, targets] -= 1.0
    grad /= batch_size
    return _node(loss, (logits, lambda g: g * grad))
