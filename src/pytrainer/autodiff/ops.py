import numpy as np
from pytrainer.autodiff.tensor import Tensor


def _tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _node(out_data, *inputs):
    """Wrap ``out_data`` as a graph node.

    ``inputs`` are ``(parent, grad_fn)`` pairs; ``grad_fn`` maps ``out.grad``
    to that parent's gradient contribution, which backward accumulates.
    """
    parents = tuple(parent for parent, _ in inputs)
    out = Tensor(
        out_data,
        requires_grad=any(p.requires_grad for p in parents),
        _parents=parents,
    )

    def backward():
        for parent, grad_fn in inputs:
            if parent.requires_grad:
                parent._ensure_grad()
                parent.grad += grad_fn(out.grad)

    out._backward = backward
    return out


def _reduction_grad(grad, shape, axis, keepdims):
    """Restore reduced dimensions and broadcast to the input shape."""
    if axis is not None and not keepdims:
        grad = np.expand_dims(grad, axis)
    return np.broadcast_to(grad, shape)


def _unbroadcast(grad, shape):
    """Sum out axes that were broadcasted to match `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


def add(a, b):
    a, b = _tensor(a), _tensor(b)
    return _node(
        a.data + b.data,
        (a, lambda g: _unbroadcast(g, a.shape)),
        (b, lambda g: _unbroadcast(g, b.shape)),
    )


def mul(a, b):
    a, b = _tensor(a), _tensor(b)
    return _node(
        a.data * b.data,
        (a, lambda g: _unbroadcast(g * b.data, a.shape)),
        (b, lambda g: _unbroadcast(g * a.data, b.shape)),
    )


def sub(a, b):
    return add(a, mul(b, -1.0))


def div(a, b):
    return mul(a, pow(_tensor(b), -1.0))


def pow(a, p):
    return _node(a.data ** p, (a, lambda g: g * p * a.data ** (p - 1)))


def exp(a):
    out_data = np.exp(a.data)
    return _node(out_data, (a, lambda g: g * out_data))


def log(a):
    return _node(np.log(a.data), (a, lambda g: g / a.data))


def sum(a, axis=None, keepdims=False):
    return _node(
        np.sum(a.data, axis=axis, keepdims=keepdims),
        (a, lambda g: _reduction_grad(g, a.shape, axis, keepdims)),
    )


def mean(a, axis=None, keepdims=False):
    out_data = np.mean(a.data, axis=axis, keepdims=keepdims)
    count = a.data.size // np.size(out_data)  # elements averaged per output
    return _node(
        out_data,
        (a, lambda g: _reduction_grad(g, a.shape, axis, keepdims) / count),
    )


def max(a, axis=None, keepdims=False):
    out_data = np.max(a.data, axis=axis, keepdims=keepdims)

    def grad_fn(g):
        # Split the gradient evenly between tied maxima.
        peak = _reduction_grad(out_data, a.shape, axis, keepdims)
        mask = (a.data == peak).astype(float)
        mask /= np.sum(mask, axis=axis, keepdims=True)
        return _reduction_grad(g, a.shape, axis, keepdims) * mask

    return _node(out_data, (a, grad_fn))


def getitem(a, indices):
    def grad_fn(g):
        full = np.zeros_like(a.data)
        np.add.at(full, indices, g)
        return full

    return _node(a.data[indices], (a, grad_fn))


def matmul(a, b):
    a, b = _tensor(a), _tensor(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"matmul supports only 2D matrices; got shapes {a.shape} and {b.shape}"
        )
    return _node(
        a.data @ b.data,
        (a, lambda g: g @ b.data.T),
        (b, lambda g: a.data.T @ g),
    )


def relu(a):
    # Convention: ReLU'(0) = 0.
    return _node(np.maximum(0.0, a.data), (a, lambda g: g * (a.data > 0)))


def tanh(a):
    out_data = np.tanh(a.data)
    return _node(out_data, (a, lambda g: g * (1.0 - out_data**2)))
