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


# ----------------------------------------------------------------------
# Forward-graph weight mask: W_effective = W * M
# ----------------------------------------------------------------------
def masked_layer(seed=SEED):
    rng = np.random.default_rng(seed)
    lin = Linear(3, 4, rng)
    mask = rng.random((3, 4)) > 0.5
    lin.set_weight_mask(mask)
    return lin, mask, rng


def test_mask_starts_dense_and_is_not_trainable():
    lin = Linear(3, 4, np.random.default_rng(SEED))
    np.testing.assert_array_equal(lin.weight_mask.data, 1.0)
    assert lin.weight_mask.requires_grad is False
    assert all(p is not lin.weight_mask for p in lin.parameters())
    assert lin.active_weights == 12


def test_forward_uses_effective_weight():
    lin, mask, rng = masked_layer()
    x = rng.standard_normal((5, 3))
    expected = x @ (lin.weight.data * mask) + lin.bias.data
    np.testing.assert_allclose(lin(x).data, expected)


def test_masked_connection_has_no_effect_whatever_its_raw_value():
    lin, mask, rng = masked_layer()
    x = rng.standard_normal((5, 3))
    before = lin(x).data.copy()
    lin.weight.data[~mask] = 1e6  # corrupt every pruned raw weight
    np.testing.assert_array_equal(lin(x).data, before)


def test_gradient_is_mask_times_effective_gradient():
    lin, mask, rng = masked_layer()
    x = rng.standard_normal((5, 3))
    g = rng.standard_normal((5, 4))
    ops.sum(ops.mul(lin(x), Tensor(g))).backward()
    # dL/dW_eff = x^T g ; dL/dW = M * dL/dW_eff
    np.testing.assert_allclose(lin.weight.grad, mask * (x.T @ g))
    assert np.all(lin.weight.grad[~mask] == 0.0)
    assert lin.weight_mask.grad is None  # constant: no gradient stored


def test_masked_gradients_match_finite_differences():
    lin, mask, rng = masked_layer()
    x = rng.standard_normal((4, 3))
    g = rng.standard_normal((4, 4))
    lin.weight.grad = None
    ops.sum(ops.mul(lin(x), Tensor(g))).backward()
    f = lambda: float(np.sum(lin(x).data * g))
    eps = 1e-6
    for i in np.ndindex(lin.weight.shape):
        old = lin.weight.data[i]
        lin.weight.data[i] = old + eps; fp = f()
        lin.weight.data[i] = old - eps; fm = f()
        lin.weight.data[i] = old
        np.testing.assert_allclose(lin.weight.grad[i], (fp - fm) / (2 * eps), atol=1e-8)


def test_set_weight_mask_is_in_place_and_validated():
    lin = Linear(3, 4, np.random.default_rng(SEED))
    mask_tensor, mask_array = lin.weight_mask, lin.weight_mask.data
    lin.set_weight_mask(np.zeros((3, 4), dtype=bool))
    assert lin.weight_mask is mask_tensor and lin.weight_mask.data is mask_array
    assert lin.active_weights == 0
    with pytest.raises(ValueError):
        lin.set_weight_mask(np.ones((4, 3)))
    with pytest.raises(ValueError):
        lin.set_weight_mask(np.full((3, 4), 0.5))


def test_dense_mask_leaves_outputs_and_gradients_bit_identical():
    rng = np.random.default_rng(SEED)
    lin = Linear(3, 4, rng)
    x = rng.standard_normal((5, 3))
    out = lin(x)
    np.testing.assert_array_equal(out.data, x @ lin.weight.data + lin.bias.data)
    g = rng.standard_normal((5, 4))
    ops.sum(ops.mul(out, Tensor(g))).backward()
    np.testing.assert_array_equal(lin.weight.grad, x.T @ g)


def test_mlp_parameters_exclude_masks():
    model = MLP(np.random.default_rng(SEED))
    masks = {id(l.weight_mask) for l in (model.fc1, model.fc2, model.fc3)}
    assert not any(id(p) in masks for p in model.parameters())
    assert len(model.parameters()) == 6