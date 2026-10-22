import numpy as np
from pytrainer.autodiff.tensor import Tensor


def _reduction_grad(upstream_grad, input_shape, axis, keepdims):
    """Restore reduced dimensions and broadcast to the input shape."""
    if axis is None:
        reduced_shape = (1,) * len(input_shape)
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)

        # Normalize negative axes against the original input dimensions.
        axes = tuple(
            ax + len(input_shape) if ax < 0 else ax
            for ax in axes
        )

        reduced_shape = tuple(
            1 if i in axes else dim
            for i, dim in enumerate(input_shape)
        )

    # Works for both keepdims settings because the target shape
    # has size 1 along every reduced dimension.
    grad = np.asarray(upstream_grad).reshape(reduced_shape)
    return np.broadcast_to(grad, input_shape)


def _unbroadcast(grad, shape):
    """Sum out axes that were broadcasted to match `shape`."""
    grad = np.asarray(grad)

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad.reshape(shape)


def add(a, b):
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)

    out_data = a.data + b.data
    requires_grad = a.requires_grad or b.requires_grad

    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        _parents=(a, b),
    )

    def backward():
        if a.requires_grad:
            a._ensure_grad()
            a.grad += _unbroadcast(out.grad, a.shape)

        if b.requires_grad:
            b._ensure_grad()
            b.grad += _unbroadcast(out.grad, b.shape)

    out._backward = backward
    return out


def mul(a, b):
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    out_data = a.data * b.data
    requires_grad = a.requires_grad or b.requires_grad

    out = Tensor(
        out_data,
        requires_grad=requires_grad,
        _parents=(a, b),
    )

    def backward():
        if a.requires_grad:
            a._ensure_grad()
            grad_a = out.grad * b.data
            a.grad += _unbroadcast(grad_a, a.shape)

        if b.requires_grad:
            b._ensure_grad()
            grad_b = out.grad * a.data
            b.grad += _unbroadcast(grad_b, b.shape)

    out._backward = backward
    return out


def sub(a, b):
    b = b if isinstance(b, Tensor) else Tensor(b)
    return add(a, mul(b, -1.0))


def div(a, b):
    b = b if isinstance(b, Tensor) else Tensor(b)
    return mul(a, pow(b, -1.0))


def pow(a, p):
    out_data = a.data ** p
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _parents=(a,),
    )

    def backward():
        if a.requires_grad:
            g = out.grad
            a._ensure_grad()
            a.grad += g * p * (a.data ** (p - 1))

    out._backward = backward
    return out


def exp(a):
    out_data = np.exp(a.data)
    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _parents=(a,)
    )

    def backward():
        if a.requires_grad:
            g = out.grad
            a._ensure_grad()
            a.grad += g * out_data

    out._backward = backward
    return out


def log(a):
    out_data = np.log(a.data)
    out = Tensor(
            out_data,
            requires_grad=a.requires_grad,
            _parents=(a,)
        )

    def backward():
        if a.requires_grad:
            g = out.grad
            a._ensure_grad()
            a.grad += g / a.data

    out._backward = backward
    return out


def sum(a, axis=None, keepdims=False):
    out_data = np.sum(a.data, axis=axis, keepdims=keepdims)

    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _parents=(a,),
    )

    def backward():
        if a.requires_grad:
            a._ensure_grad()

            grad = _reduction_grad(
                out.grad, a.shape, axis, keepdims
            )

            a.grad += grad

    out._backward = backward
    return out


def max(a, axis=None, keepdims=False):
    out_data = np.max(a.data, axis=axis, keepdims=keepdims)

    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _parents=(a,),
    )

    def backward():
        if a.requires_grad:
            g = out.grad
            a._ensure_grad()

            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
                out_data_exp = np.expand_dims(out_data, axis=axis)
            else:
                out_data_exp = out_data

            mask = (a.data == out_data_exp).astype(float)
            count = np.sum(mask, axis=axis, keepdims=True)
            mask = mask / np.maximum(count, 1)

            a.grad += np.broadcast_to(g * mask, a.shape)

    out._backward = backward
    return out


def getitem(a, indices):
    out_data = a.data[indices]

    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _parents=(a,),
    )

    def backward():
        if a.requires_grad:
            g = out.grad
            a._ensure_grad()
            np.add.at(a.grad, indices, g)

    out._backward = backward
    return out


def matmul(a, b):
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError(
            "matmul currently supports only 2D matrices; "
            f"got shapes {a.shape} and {b.shape}"
        )

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            "matmul dimension mismatch: "
            f"cannot multiply shapes {a.shape} and {b.shape}"
        )

    out_data = a.data @ b.data

    out = Tensor(
        out_data,
        requires_grad=a.requires_grad or b.requires_grad,
        _parents=(a, b),
    )

    def backward():
        g = out.grad

        if a.requires_grad:
            a._ensure_grad()
            a.grad += g @ b.data.T

        if b.requires_grad:
            b._ensure_grad()
            b.grad += a.data.T @ g

    out._backward = backward
    return out


def mean(a, axis=None, keepdims=False):
    out_data = np.mean(a.data, axis=axis, keepdims=keepdims)

    if axis is None:
        count = a.data.size
    else:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)

        # Normalize negative axes for counting reduced elements.
        axes = tuple(
            ax + a.data.ndim if ax < 0 else ax
            for ax in axes
        )
        count = int(np.prod([a.shape[ax] for ax in axes]))

    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _parents=(a,),
    )

    def backward():
        if a.requires_grad:
            a._ensure_grad()
            grad = _reduction_grad(
                out.grad, a.shape, axis, keepdims
            )
            a.grad += grad / count

    out._backward = backward
    return out



def relu(a):

    out_data = np.maximum(0.0, a.data)

    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _parents=(a,),
    )

    def backward():
        if a.requires_grad:
            a._ensure_grad()

            # Convention: ReLU'(0) = 0.
            local_grad = (a.data > 0).astype(a.data.dtype)
            a.grad += out.grad * local_grad

    out._backward = backward
    return out


def tanh(a):

    # Save the forward output for the derivative.
    out_data = np.tanh(a.data)

    out = Tensor(
        out_data,
        requires_grad=a.requires_grad,
        _parents=(a,),
    )

    def backward():
        if a.requires_grad:
            a._ensure_grad()

            # tanh'(x) = 1 - tanh(x)^2
            local_grad = 1.0 - out.data**2
            a.grad += out.grad * local_grad

    out._backward = backward
    return out
