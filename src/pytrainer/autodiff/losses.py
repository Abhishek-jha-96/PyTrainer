
import numpy as np

from pytrainer.autodiff.tensor import Tensor


def softmax_cross_entropy(logits, targets):
    """
    Numerically stable mean softmax cross-entropy.

    Parameters
    ----------
    logits : Tensor
        Raw class scores with shape (N, C).
    targets : array-like
        Either:
        - Integer class indices with shape (N,), values in [0, C).
        - One-hot labels with shape (N, C), one active class per row.

    Returns
    -------
    Tensor
        Scalar mean cross-entropy loss.

    Notes
    -----
    Only logits receive gradients. Targets are labels, not differentiable
    inputs. The gradient for mean reduction is (softmax(logits) - targets) / N.
    """
    if not isinstance(logits, Tensor):
        raise TypeError("logits must be a Tensor")

    if logits.data.ndim != 2:
        raise ValueError(
            f"logits must have shape (N, C); got {logits.shape}"
        )

    batch_size, num_classes = logits.shape

    if batch_size == 0 or num_classes == 0:
        raise ValueError("logits must have non-empty batch and class dimensions")

    targets = np.asarray(targets)

    # Convert integer class indices into one-hot targets.
    if targets.ndim == 1:
        if targets.shape != (batch_size,):
            raise ValueError(
                f"class-index targets must have shape ({batch_size},); "
                f"got {targets.shape}"
            )

        if not np.issubdtype(targets.dtype, np.integer):
            raise TypeError("class-index targets must contain integers")

        if np.any(targets < 0) or np.any(targets >= num_classes):
            raise ValueError("target class indices are out of range")

        target_indices = targets.astype(np.intp, copy=False)
        target_matrix = np.zeros(
            (batch_size, num_classes), dtype=logits.data.dtype
        )
        target_matrix[np.arange(batch_size), target_indices] = 1.0

    elif targets.ndim == 2:
        if targets.shape != (batch_size, num_classes):
            raise ValueError(
                f"one-hot targets must have shape "
                f"({batch_size}, {num_classes}); got {targets.shape}"
            )

        if not np.issubdtype(targets.dtype, np.number):
            raise TypeError("one-hot targets must be numeric")

        target_matrix = targets.astype(logits.data.dtype, copy=False)

        if not np.all(np.isfinite(target_matrix)):
            raise ValueError("one-hot targets must be finite")

        if np.any(target_matrix < 0) or not np.allclose(
            target_matrix.sum(axis=1), 1.0
        ):
            raise ValueError(
                "each one-hot target row must be non-negative "
                "and sum to 1"
            )

    else:
        raise ValueError(
            "targets must have shape (N,) for class indices "
            "or (N, C) for one-hot labels"
        )

    # Stable log-softmax via log-sum-exp.
    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    log_partition = np.log(
        np.sum(np.exp(shifted), axis=1, keepdims=True)
    )
    log_probs = shifted - log_partition
    probs = np.exp(log_probs)

    loss_data = -np.sum(target_matrix * log_probs) / batch_size

    out = Tensor(
        loss_data,
        requires_grad=logits.requires_grad,
        _parents=(logits,),
    )

    def backward():
        if logits.requires_grad:
            logits._ensure_grad()
            logits.grad += (
                out.grad
                * (probs - target_matrix)
                / batch_size
            )

    out._backward = backward
    return out
