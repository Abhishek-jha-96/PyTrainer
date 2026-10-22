
import numpy as np
import pytest

from pytrainer.autodiff.tensor import Tensor
from pytrainer.autodiff import ops


def test_relu_positive_and_negative_inputs():
    x = Tensor([-2.0, -1.0, 1.0, 3.0], requires_grad=True)

    y = ops.relu(x)

    np.testing.assert_allclose(y.data, [0.0, 0.0, 1.0, 3.0])

    y.backward(np.ones(4))

    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0, 1.0])


def test_relu_zero_derivative_convention():
    # Documented convention: ReLU'(0) = 0.
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)

    y = ops.relu(x)
    y.backward(np.ones(3))

    np.testing.assert_allclose(y.data, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(x.grad, [0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "values",
    [
        [-3.0, -1.0, 0.0, 1.0, 3.0],
        [-0.5, 0.25, 2.0],
    ],
)
def test_tanh_forward_and_backward(values):
    data = np.array(values)
    x = Tensor(data, requires_grad=True)

    y = ops.tanh(x)

    np.testing.assert_allclose(y.data, np.tanh(data))
    y.backward(np.ones_like(data))

    expected_grad = 1.0 - np.tanh(data) ** 2
    np.testing.assert_allclose(x.grad, expected_grad)


def test_tanh_finite_difference():
    data = np.array([-2.0, -0.5, 0.25, 1.5])
    upstream = np.array([0.5, -1.0, 2.0, 0.75])
    eps = 1e-6

    x = Tensor(data.copy(), requires_grad=True)
    y = ops.tanh(x)
    y.backward(upstream)

    numerical = (
        np.tanh(data + eps) - np.tanh(data - eps)
    ) / (2 * eps)
    numerical *= upstream

    np.testing.assert_allclose(
        x.grad, numerical, rtol=1e-5, atol=1e-6
    )


def test_relu_finite_difference_away_from_zero():
    # Avoid zero, where ReLU is not differentiable.
    data = np.array([-2.0, -0.5, 0.5, 2.0])
    eps = 1e-6

    x = Tensor(data.copy(), requires_grad=True)
    y = ops.relu(x)
    y.backward(np.ones_like(data))

    numerical = (
        np.maximum(0.0, data + eps)
        - np.maximum(0.0, data - eps)
    ) / (2 * eps)

    np.testing.assert_allclose(
        x.grad, numerical, atol=1e-6
    )
