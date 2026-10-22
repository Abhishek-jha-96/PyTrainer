import numpy as np

from pytrainer.autodiff.tensor import Tensor
from pytrainer.autodiff import ops
from pytrainer.tests.gradient_check import gradcheck


def make_tensor(data):
    return Tensor(data, requires_grad=True)


def test_add():
    a = make_tensor([0.7, 1.2, -0.4])
    b = make_tensor([1.1, -0.3, 0.8])

    assert gradcheck(lambda a, b: ops.sum(a + b), (a, b))


def test_sub():
    a = make_tensor([0.7, 1.2, -0.4])
    b = make_tensor([1.1, -0.3, 0.8])

    assert gradcheck(lambda a, b: ops.sum(a - b), (a, b))


def test_mul():
    a = make_tensor([0.7, 1.2, -0.4])
    b = make_tensor([1.1, -0.3, 0.8])

    assert gradcheck(lambda a, b: ops.sum(a * b), (a, b))


def test_div():
    a = make_tensor([0.7, 1.2, -0.4])
    b = make_tensor([1.1, 1.5, 0.8])

    assert gradcheck(lambda a, b: ops.sum(a / b), (a, b))


def test_pow():
    # Positive bases avoid numerical issues for non-integer exponents.
    a = make_tensor([0.7, 1.2, 1.8])

    assert gradcheck(lambda a: ops.sum(a ** 2.5), (a,))


def test_broadcasting():
    a = make_tensor([[0.2, 0.5, 0.9], [1.1, 1.4, 1.8]])
    b = make_tensor([0.3, 0.6, 1.0])

    def loss(a, b):
        return ops.mean(a * b + b)

    assert gradcheck(loss, (a, b))


def test_sum_axis():
    a = make_tensor([[0.2, 0.5, 0.9], [1.1, 1.4, 1.8]])

    # Sum along axis 0 produces a vector; reduce again to a scalar.
    assert gradcheck(
        lambda a: ops.sum(ops.sum(a, axis=0)),
        (a,),
    )


def test_mean():
    a = make_tensor([[0.2, 0.5, 0.9], [1.1, 1.4, 1.8]])

    assert gradcheck(lambda a: ops.mean(a), (a,))


def test_relu():
    # Avoid zero, where ReLU is not differentiable.
    a = make_tensor([-1.2, -0.4, 0.3, 1.1])

    assert gradcheck(lambda a: ops.sum(ops.relu(a) ** 2), (a,))


def test_tanh():
    a = make_tensor([-1.2, -0.4, 0.3, 1.1])

    assert gradcheck(lambda a: ops.sum(ops.tanh(a)), (a,))


def test_matmul():
    a = make_tensor([[0.2, 0.5, 0.9], [1.1, 1.4, 1.8]])
    b = make_tensor([[0.3, 0.6], [0.8, 1.0], [1.2, 1.5]])

    assert gradcheck(lambda a, b: ops.sum(a @ b), (a, b))


def test_combined_expression():
    a = make_tensor([[0.4, 0.7], [1.0, 1.3]])
    b = make_tensor([0.2, 0.6])

    def loss(a, b):
        z = a * b + a
        z = ops.tanh(z)
        return ops.mean(z ** 2)

    assert gradcheck(loss, (a, b))


def test_shared_subgraph():
    a = make_tensor([0.4, 0.8, 1.2])

    def loss(a):
        shared = a * a
        return ops.sum(shared + shared)

    assert gradcheck(loss, (a,))