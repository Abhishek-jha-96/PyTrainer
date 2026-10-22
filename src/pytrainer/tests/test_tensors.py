import numpy as np
from pytrainer.autodiff.tensor import Tensor


def test_tensor_constructions():
    """Acceptance test: tensor construction preserves shape and values."""
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    t = Tensor(data, requires_grad=True)

    print(t._graph_metadata())

    assert t.shape == (2, 2), f"Expected shape (2, 2), got {t.shape}"
    assert np.array_equal(t.data, data), "Data values not preserved"
    assert t.requires_grad is True
    assert t.grad is None

    print("Tensor construction tested.")


def test_backward():
    x = Tensor(3.0, requires_grad=True)
    y = x * x + x

    y.backward()

    assert np.allclose(x.grad, 7.0)
    print("x * x + x:", x.grad)

def test_backward_with_subgraph():
    x = Tensor(3.0, requires_grad=True)

    shared = x * x
    y = shared + shared

    y.backward()

    assert np.allclose(x.grad, 12.0)
    print("Shared subgraph:", x.grad)

def test_unbroadcast():
    N, D = 4, 3

    A = Tensor(np.ones((N, D)), requires_grad=True)
    b = Tensor(np.ones(D), requires_grad=True)

    (A + b).backward(seed=np.ones((N, D)))
    assert np.allclose(A.grad, 1.0)
    assert np.allclose(b.grad, N)

    print("All unbroadcast tests passed.")

def test_arithmetic_gradients():
    # Addition: y = x + 3, dy/dx = 1
    x = Tensor(2.0, requires_grad=True)
    (x + 3).backward()
    assert np.allclose(x.grad, 1.0)

    # Subtraction: y = x - 3, dy/dx = 1
    x = Tensor(2.0, requires_grad=True)
    (x - 3).backward()
    assert np.allclose(x.grad, 1.0)

    # Multiplication: y = x * x, dy/dx = 2x
    x = Tensor(3.0, requires_grad=True)
    (x * x).backward()
    assert np.allclose(x.grad, 6.0)

    # Division: y = x / 2, dy/dx = 1/2
    x = Tensor(6.0, requires_grad=True)
    (x / 2).backward()
    assert np.allclose(x.grad, 0.5)

    # Denominator derivative: y = 6 / x, dy/dx = -6/x^2
    x = Tensor(2.0, requires_grad=True)
    (6 / x).backward()
    assert np.allclose(x.grad, -1.5)

    # Reverse subtraction: y = 10 - x
    x = Tensor(3.0, requires_grad=True)
    (10 - x).backward()
    assert np.allclose(x.grad, -1.0)

    # Reverse division: y = 10 / x
    x = Tensor(4.0, requires_grad=True)
    (10 / x).backward()
    assert np.allclose(x.grad, -0.625)

    print("Arithmetic gradient tests passed.")


if __name__ == "__main__":
    test_tensor_constructions()
    test_backward()
    test_backward_with_subgraph()
    test_unbroadcast()
    test_arithmetic_gradients()
    print("All tests passed.")