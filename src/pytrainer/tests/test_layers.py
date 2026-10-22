"""Tests for Linear / MLP shapes, gradients and weight initialization."""

import numpy as np
import pytest

from pytrainer.autodiff import ops
from pytrainer.autodiff.tensor import Tensor
from pytrainer.nn.layers import Linear, he_normal, xavier_uniform
from pytrainer.nn.models import MLP

SEED = 0


# ----------------------------------------------------------------------
# Linear: shapes, forward, gradients
# ----------------------------------------------------------------------
def test_linear_shapes_and_forward():
    rng = np.random.default_rng(SEED)
    lin = Linear(3, 4, rng)
    x = rng.standard_normal((5, 3))
    y = lin(x)
    assert lin.weight.shape == (3, 4)
    assert lin.bias.shape == (4,)
    assert y.shape == (5, 4)
    np.testing.assert_allclose(y.data, x @ lin.weight.data + lin.bias.data)


def test_linear_gradients():
    rng = np.random.default_rng(SEED)
    lin = Linear(3, 4, rng)
    x = rng.standard_normal((5, 3))
    g = rng.standard_normal((5, 4))
    ops.sum(ops.mul(lin(x), Tensor(g))).backward()
    assert lin.bias.grad.shape == (4,)
    np.testing.assert_allclose(lin.weight.grad, x.T @ g)
    np.testing.assert_allclose(lin.bias.grad, g.sum(axis=0))


def test_linear_rejects_bad_input_shape():
    lin = Linear(3, 4, np.random.default_rng(SEED))
    with pytest.raises(ValueError):
        lin(np.ones(3))
    with pytest.raises(ValueError):
        lin(np.ones((2, 5)))


def test_linear_rejects_unknown_init():
    with pytest.raises(ValueError):
        Linear(3, 4, np.random.default_rng(SEED), init="zeros")


# ----------------------------------------------------------------------
# Initializers: statistics follow the formulas on (fan_in, fan_out) layout
# ----------------------------------------------------------------------
def test_he_normal_std_uses_fan_in_of_storage_layout():
    # fan_in != fan_out so swapping them would be caught.
    fan_in, fan_out = 400, 100
    w = he_normal((fan_in, fan_out), np.random.default_rng(SEED))
    assert w.shape == (fan_in, fan_out)
    assert abs(w.mean()) < 0.01
    np.testing.assert_allclose(w.std(), np.sqrt(2.0 / fan_in), rtol=0.02)


def test_xavier_uniform_bounds_and_variance():
    fan_in, fan_out = 400, 100
    a = np.sqrt(6.0 / (fan_in + fan_out))
    w = xavier_uniform((fan_in, fan_out), np.random.default_rng(SEED))
    assert w.shape == (fan_in, fan_out)
    assert np.all(np.abs(w) <= a)
    # Var of U(-a, a) is a^2 / 3 = 2 / (fan_in + fan_out)
    np.testing.assert_allclose(w.var(), a**2 / 3, rtol=0.03)


# ----------------------------------------------------------------------
# MLP: policy, biases, symmetry, reproducibility, finiteness
# ----------------------------------------------------------------------
def test_mlp_init_policy():
    model = MLP(np.random.default_rng(SEED))
    assert model.fc1.init == "he_normal"
    assert model.fc2.init == "he_normal"
    assert model.fc3.init == "xavier_uniform"


def test_mlp_biases_are_zero_and_weights_are_not():
    model = MLP(np.random.default_rng(SEED))
    for layer in (model.fc1, model.fc2, model.fc3):
        np.testing.assert_array_equal(layer.bias.data, 0.0)
        assert np.any(layer.weight.data != 0.0)
        # No two hidden units start identical (symmetry is broken).
        cols = layer.weight.data.T
        assert len(np.unique(cols, axis=0)) == cols.shape[0]


def test_mlp_init_is_reproducible_with_one_seeded_generator():
    a = MLP(np.random.default_rng(SEED))
    b = MLP(np.random.default_rng(SEED))
    c = MLP(np.random.default_rng(SEED + 1))
    for pa, pb in zip(a.parameters(), b.parameters()):
        np.testing.assert_array_equal(pa.data, pb.data)
    assert any(
        not np.array_equal(pa.data, pc.data)
        for pa, pc in zip(a.parameters(), c.parameters())
    )


def test_initial_parameters_and_logits_are_finite():
    rng = np.random.default_rng(SEED)
    model = MLP(rng)
    for p in model.parameters():
        assert np.all(np.isfinite(p.data))
    logits = model(rng.standard_normal((64, 2)))
    assert logits.shape == (64, 2)
    assert np.all(np.isfinite(logits.data))


def test_mlp_param_count():
    model = MLP(np.random.default_rng(SEED))
    total = sum(p.data.size for p in model.parameters())
    assert total == (2 * 32 + 32) + (32 * 16 + 16) + (16 * 2 + 2)