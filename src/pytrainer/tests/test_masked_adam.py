"""Mask-aware Adam: pruned entries stay exactly zero on every step.

Invariant checked after every update:
    weights == 0, gradients == 0, m == 0 and v == 0 at pruned entries,
    and the mask itself is never optimized.
"""

import numpy as np
import pytest

from pytrainer.autodiff.tensor import Tensor
from pytrainer.nn.optim import Adam, MaskedAdam
from pytrainer.nn.pruning import GlobalTaylorPruner

SHAPES = {"fc1.weight": (2, 32), "fc2.weight": (32, 16), "fc3.weight": (16, 2)}


def param(value):
    return Tensor(np.array(value, dtype=np.float64), requires_grad=True)


def assert_invariant(opt, p, mask):
    m, v = opt.moments(p)
    assert np.all(p.data[~mask] == 0.0)
    if p.grad is not None:
        assert np.all(np.asarray(p.grad)[~mask] == 0.0)
    assert np.all(m[~mask] == 0.0) and np.all(v[~mask] == 0.0)


def build_with_momentum(cls, steps=20, seed=0):
    """A parameter whose every entry has built up nonzero momentum."""
    rng = np.random.default_rng(seed)
    p = param(rng.normal(size=(4, 5)))
    opt = cls([p])
    for _ in range(steps):
        p.grad = rng.normal(size=p.data.shape) + 1.0  # biased, so m != 0
        opt.step()
    assert np.all(opt.m[0] != 0)
    return p, opt, rng


MASK = np.array([[True, False, True, False, True]] * 4)


# ------------------------------------------------------ the problem itself
def test_plain_adam_revives_a_zeroed_weight_through_momentum():
    p, opt, _ = build_with_momentum(Adam)
    p.data[~MASK] = 0.0  # enforce once, momentum left in place
    p.grad = np.zeros_like(p.data)  # g_t = 0 everywhere
    opt.step()
    assert np.all(p.data[~MASK] != 0.0)  # m_{t-1} moved them anyway


# --------------------------------------------------------------- mask_state
def test_mask_state_clears_only_pruned_moments():
    p, opt, _ = build_with_momentum(Adam)
    m_before, v_before = opt.m[0].copy(), opt.v[0].copy()
    opt.mask_state(p, MASK)
    m, v = opt.moments(p)
    assert np.all(m[~MASK] == 0) and np.all(v[~MASK] == 0)
    np.testing.assert_array_equal(m[MASK], m_before[MASK])
    np.testing.assert_array_equal(v[MASK], v_before[MASK])
    assert m is opt.m[0] and v is opt.v[0]  # in place, same state arrays


def test_mask_state_then_zero_gradients_keeps_weights_at_zero():
    p, opt, _ = build_with_momentum(Adam)
    p.data[~MASK] = 0.0
    opt.mask_state(p, MASK)
    for _ in range(200):
        p.grad = np.where(MASK, 0.3, 0.0)  # forward mask gives exact zeros
        opt.step()
        assert_invariant(opt, p, MASK)
    assert np.all(p.data[MASK] != 0)  # kept entries still train


def test_mask_state_accepts_float_masks_and_validates():
    p, opt, _ = build_with_momentum(Adam)
    opt.mask_state(p, MASK.astype(np.float64))
    assert np.all(opt.m[0][~MASK] == 0)
    with pytest.raises(ValueError, match="shape"):
        opt.mask_state(p, np.ones((5, 4), dtype=bool))
    with pytest.raises(ValueError, match="0 and 1"):
        opt.mask_state(p, np.full((4, 5), 0.5))
    with pytest.raises(ValueError, match="not managed"):
        opt.mask_state(param(np.ones((4, 5))), MASK)


# --------------------------------------------------------------- MaskedAdam
def test_masked_adam_holds_invariant_under_hostile_gradients():
    p, opt, rng = build_with_momentum(MaskedAdam)
    opt.set_mask(p, MASK)
    assert_invariant(opt, p, MASK)
    for _ in range(500):
        p.grad = rng.normal(size=p.data.shape) * 10.0  # nonzero at pruned entries
        opt.step()
        assert_invariant(opt, p, MASK)


def test_masked_adam_matches_plain_adam_on_kept_entries():
    rng = np.random.default_rng(4)
    init = rng.normal(size=(4, 5))
    a, b = param(init.copy()), param(init.copy())
    plain, masked = Adam([a]), MaskedAdam([b])
    masked.set_mask(b, MASK)
    for _ in range(100):
        g = rng.normal(size=init.shape)
        a.grad = np.where(MASK, g, 0.0)
        b.grad = g.copy()
        plain.step()
        masked.step()
    np.testing.assert_array_equal(b.data[MASK], a.data[MASK])


def test_unmasked_parameters_are_plain_adam():
    rng = np.random.default_rng(5)
    init = rng.normal(size=3)
    a, b = param(init.copy()), param(init.copy())
    plain, masked = Adam([a]), MaskedAdam([b])
    for _ in range(50):
        g = rng.normal(size=3)
        a.grad, b.grad = g.copy(), g.copy()
        plain.step()
        masked.step()
    np.testing.assert_array_equal(a.data, b.data)
    assert masked.mask_of(b) is None


def test_stored_mask_is_never_optimized_or_aliased():
    p, opt, rng = build_with_momentum(MaskedAdam)
    caller_mask = MASK.copy()
    opt.set_mask(p, caller_mask)
    caller_mask[:] = True  # caller mutates its array afterwards
    snapshot = opt.mask_of(p).copy()
    for _ in range(50):
        p.grad = rng.normal(size=p.data.shape)
        opt.step()
    np.testing.assert_array_equal(opt.mask_of(p), snapshot)
    np.testing.assert_array_equal(snapshot, MASK)


def test_a_mask_tensor_cannot_be_registered_as_a_parameter():
    mask_tensor = Tensor(np.ones((4, 5)))  # requires_grad=False, like weight_mask
    with pytest.raises(ValueError, match="require grad"):
        MaskedAdam([param(np.ones((4, 5))), mask_tensor])


# ------------------------------------------------- pruner + optimizer loop
def run_loop(optimizer_cls, hostile, epochs=12, seed=0):
    """Mirror of the training loop: observe -> step -> enforce, prune per epoch."""
    rng = np.random.default_rng(seed)
    weights = {n: param(rng.normal(size=s)) for n, s in SHAPES.items()}
    masks = {n: Tensor(np.ones(s)) for n, s in SHAPES.items()}
    biases = [param(np.zeros(s[1])) for s in SHAPES.values()]
    opt = optimizer_cls(list(weights.values()) + biases)
    pruner = GlobalTaylorPruner(weights, masks, output_name="fc3.weight")
    for epoch in range(1, epochs + 1):
        for _ in range(5):
            opt.zero_grad()
            for n, w in weights.items():
                g = rng.normal(size=w.data.shape)
                w.grad = g if hostile else g * masks[n].data  # engine: M * dL/dW_eff
            for b in biases:
                b.grad = rng.normal(size=b.data.shape)
            pruner.observe()
            opt.step()
            pruner.enforce(opt)
            pruner.check_integrity(opt)
            for n, w in weights.items():
                assert_invariant(opt, w, pruner.mask(n))
        if epoch > 2:
            pruner.prune_to(min(0.9, 0.1 * (epoch - 2)), optimizer=opt)
            pruner.check_integrity(opt)
    return weights, masks, biases, opt, pruner


@pytest.mark.parametrize("cls", [MaskedAdam, Adam])
@pytest.mark.parametrize("hostile", [False, True])
def test_invariant_holds_after_every_update_in_the_loop(cls, hostile):
    _, masks, biases, _, pruner = run_loop(cls, hostile)
    assert round(pruner.sparsity() * 608) == 547
    assert all(np.any(b.data != 0) for b in biases)  # biases still train
    for n, m in masks.items():
        assert set(np.unique(m.data)) <= {0.0, 1.0}  # mask holds only 0/1


def test_check_integrity_catches_stale_moments():
    weights, _, _, opt, pruner = run_loop(MaskedAdam, hostile=False, epochs=4)
    name = "fc2.weight"
    dead = tuple(np.argwhere(~pruner.mask(name))[0])
    opt.moments(weights[name])[0][dead] = 1e-3  # momentum leaks back in
    with pytest.raises(RuntimeError, match="moments"):
        pruner.check_integrity(opt)


def test_check_integrity_catches_optimizer_mask_drift():
    weights, _, _, opt, pruner = run_loop(MaskedAdam, hostile=False, epochs=4)
    name = "fc2.weight"
    drifted = opt.mask_of(weights[name]).copy()
    drifted.reshape(-1)[0] = not drifted.reshape(-1)[0]
    opt.masks[opt._slot(weights[name])] = drifted
    with pytest.raises(RuntimeError, match="differs"):
        pruner.check_integrity(opt)