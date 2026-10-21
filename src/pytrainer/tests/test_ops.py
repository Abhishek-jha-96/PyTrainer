
import numpy as np
import pytest

from pytrainer.autodiff.tensor import Tensor
from pytrainer.autodiff import ops


@pytest.mark.parametrize(
    "operation,axis,keepdims,expected_shape",
    [
        ("sum", None, False, ()),
        ("sum", None, True, (1, 1)),
        ("sum", 0, False, (3,)),
        ("sum", 0, True, (1, 3)),
        ("sum", 1, False, (2,)),
        ("sum", 1, True, (2, 1)),
        ("sum", (0, 1), False, ()),
        ("sum", (0, 1), True, (1, 1)),
        ("mean", None, False, ()),
        ("mean", None, True, (1, 1)),
        ("mean", 0, False, (3,)),
        ("mean", 0, True, (1, 3)),
        ("mean", 1, False, (2,)),
        ("mean", 1, True, (2, 1)),
        ("mean", (0, 1), False, ()),
        ("mean", (0, 1), True, (1, 1)),
    ],
)
def test_reduction_shapes_and_gradients(
    operation, axis, keepdims, expected_shape
):
    data = np.arange(1.0, 7.0).reshape(2, 3)
    x = Tensor(data, requires_grad=True)

    result = getattr(ops, operation)(
        x, axis=axis, keepdims=keepdims
    )

    assert result.shape == expected_shape

    result.backward(np.ones(expected_shape) if expected_shape else None)

    assert x.grad.shape == x.shape
    assert np.all(np.isfinite(x.grad))


def test_sum_gradient():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    ops.sum(x, axis=0).backward(np.array([2.0, 3.0]))

    np.testing.assert_allclose(
        x.grad,
        [[2.0, 3.0], [2.0, 3.0]],
    )


def test_mean_gradient():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    ops.mean(x, axis=0).backward(np.array([2.0, 4.0]))

    np.testing.assert_allclose(
        x.grad,
        [[1.0, 2.0], [1.0, 2.0]],
    )


def test_mean_over_multiple_axes():
    x = Tensor(
        np.arange(1.0, 13.0).reshape(2, 3, 2),
        requires_grad=True,
    )

    result = ops.mean(x, axis=(0, 2))
    assert result.shape == (3,)

    result.backward(np.ones(3))

    np.testing.assert_allclose(
        x.grad,
        np.full(x.shape, 1.0 / 4.0),
    )


def test_negative_axis():
    x = Tensor(np.ones((2, 3)), requires_grad=True)

    ops.mean(x, axis=-1).backward(np.ones(2))

    np.testing.assert_allclose(
        x.grad,
        np.full((2, 3), 1.0 / 3.0),
    )

