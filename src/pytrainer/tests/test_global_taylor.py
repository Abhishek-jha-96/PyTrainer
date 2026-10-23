import numpy as np
import pytest

from pytrainer.nn.pruning import GlobalTaylorPruner, cubic_sparsity, removal_budget

SHAPES = {"fc1": (2, 32), "fc2": (32, 16), "fc3": (16, 2)}  # 608 weights


class T:
    """Stand-in for pytrainer Tensor: the pruner only touches .data and .grad."""

    def __init__(self, data, grad=None):
        self.data = data
        self.grad = grad


class RecordingOptimizer:
    def __init__(self):
        self.calls = []

    def set_mask(self, param, mask):
        self.calls.append((param, mask.copy()))


def make(seed=0, order="C", **kwargs):
    rng = np.random.default_rng(seed)
    weights, masks = {}, {}
    for name, shape in SHAPES.items():
        weights[name] = T(np.asarray(rng.normal(size=shape), order=order))
        masks[name] = T(np.ones(shape, order=order))
    return weights, masks, GlobalTaylorPruner(weights, masks, **kwargs)


def observe_random(weights, pruner, seed=1, steps=5):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        for p in weights.values():
            p.grad = rng.normal(size=p.data.shape)
        pruner.observe()


# ---------------------------------------------------------------- schedule
def test_cubic_schedule_values():
    assert cubic_sparsity(0, 10, 80) == 0.0
    assert cubic_sparsity(10, 10, 80) == 0.0
    assert cubic_sparsity(80, 10, 80) == 0.9
    assert cubic_sparsity(200, 10, 80) == 0.9
    assert cubic_sparsity(45, 10, 80) == pytest.approx(0.9 * (1 - 0.5 ** 3))
    s = [cubic_sparsity(e, 10, 80) for e in range(0, 101)]
    assert all(b >= a for a, b in zip(s, s[1:]))


@pytest.mark.parametrize("args", [(5, 10, 10), (5, 10, 80, 1.0), (5, 10, 80, -0.1)])
def test_cubic_schedule_rejects_bad_args(args):
    with pytest.raises(ValueError):
        cubic_sparsity(*args)


# ------------------------------------------------------------------ budget
def test_exact_global_budget_at_90_percent():
    weights, masks, pruner = make(min_keep_fraction=0.0)
    observe_random(weights, pruner)
    pruner.prune_to(0.9)
    removed = sum(int((m.data == 0).sum()) for m in masks.values())
    assert removed == removal_budget(0.9, 608) == 547
    assert sum(r["kept"] for r in pruner.layer_report().values()) == 61
    assert pruner.sparsity() == 547 / 608


@pytest.mark.parametrize("s", np.linspace(0.0, 0.95, 40))
def test_budget_exact_for_every_step_of_a_schedule(s):
    weights, masks, pruner = make(min_keep_fraction=0.0)
    observe_random(weights, pruner)
    pruner.prune_to(s)
    assert round(pruner.sparsity() * 608) == removal_budget(s, 608)


def test_global_ranking_allocates_unevenly():
    weights, masks, pruner = make(min_keep_fraction=0.0)
    observe_random(weights, pruner)
    pruner.scores["fc2"] *= 1e-6  # fc2 looks unimportant
    pruner.prune_to(0.5)
    kept = pruner.layer_report()
    assert kept["fc1"]["kept"] == 64 and kept["fc3"]["kept"] == 32
    assert kept["fc2"]["kept"] == 608 - removal_budget(0.5, 608) - 96


def test_lowest_scores_are_removed():
    weights, masks, pruner = make(min_keep_fraction=0.0)
    observe_random(weights, pruner)
    all_scores = np.concatenate([pruner.saliency(n).ravel() for n in SHAPES])
    pruner.prune_to(0.3)
    kept = np.concatenate([pruner.mask(n).ravel() for n in SHAPES])
    assert all_scores[~kept].max() <= all_scores[kept].min()


# ------------------------------------------------------------ monotonicity
def test_masks_only_shrink_and_sparsity_cannot_decrease():
    weights, masks, pruner = make()
    observe_random(weights, pruner)
    pruner.prune_to(0.3)
    before = pruner.masks
    observe_random(weights, pruner, seed=9)
    pruner.prune_to(0.6)
    for name in SHAPES:
        assert not np.any(pruner.mask(name) & ~before[name])
    with pytest.raises(ValueError):
        pruner.prune_to(0.5)


def test_resume_honours_existing_mask_tensors():
    weights, masks, _ = make()
    masks["fc2"].data[:10, :] = 0.0  # 160 already pruned
    pruner = GlobalTaylorPruner(weights, masks)
    assert pruner.sparsity() == 160 / 608
    observe_random(weights, pruner)
    pruner.prune_to(0.5)
    assert np.all(masks["fc2"].data[:10, :] == 0.0)
    assert round(pruner.sparsity() * 608) == removal_budget(0.5, 608)


def test_non_contiguous_mask_storage_is_written():
    weights, masks, pruner = make(order="F", min_keep_fraction=0.0)
    observe_random(weights, pruner)
    pruner.prune_to(0.9)
    assert sum(int((m.data == 0).sum()) for m in masks.values()) == 547


# ---------------------------------------------------------------- saliency
def test_opposite_gradients_do_not_cancel():
    w, m = T(np.ones((1, 1))), T(np.ones((1, 1)))
    p = GlobalTaylorPruner({"a": w}, {"a": m}, min_keep_fraction=0.0)
    for g in (1.0, -1.0):
        w.grad = np.full((1, 1), g)
        p.observe()
    assert p.saliency("a")[0, 0] == pytest.approx(1.0)


def test_bias_correction_first_step_equals_current_score():
    w, m = T(np.full((1, 2), 2.0)), T(np.ones((1, 2)))
    p = GlobalTaylorPruner({"a": w}, {"a": m})
    w.grad = np.array([[3.0, -0.5]])
    p.observe()
    np.testing.assert_allclose(p.saliency("a"), [[6.0, 1.0]])


def test_taylor_sq_and_magnitude_criteria():
    w, m = T(np.array([[2.0, -4.0]])), T(np.ones((1, 2)))
    p = GlobalTaylorPruner({"a": w}, {"a": m}, criterion="taylor_sq")
    w.grad = np.array([[3.0, 1.0]])
    p.observe()
    np.testing.assert_allclose(p.saliency("a"), [[36.0, 16.0]])
    q = GlobalTaylorPruner({"a": w}, {"a": m}, criterion="magnitude")
    np.testing.assert_allclose(q.saliency("a"), [[2.0, 4.0]])


def test_skips_missing_grad_and_rejects_non_finite():
    weights, masks, pruner = make()
    pruner.observe()  # all grads None
    assert not any(pruner._steps.values())
    weights["fc1"].grad = np.full(SHAPES["fc1"], np.nan)
    with pytest.raises(FloatingPointError, match="fc1"):
        pruner.observe()


def test_pruning_without_observation_raises():
    _, _, pruner = make()
    with pytest.raises(ValueError, match="observe"):
        pruner.prune_to(0.5)
    pruner.prune_to(0.0)  # nothing to remove is fine


# -------------------------------------------------------------- safeguards
def test_layer_floor_and_safeguard_diagnostic():
    weights, masks, pruner = make(min_keep_fraction=0.05)
    observe_random(weights, pruner)
    pruner.scores["fc2"] *= 1e-6
    pruner.prune_to(0.9)
    assert pruner.layer_report()["fc2"]["kept"] >= 26  # ceil(0.05 * 512)
    assert pruner.safeguard_kept > 0
    assert round(pruner.sparsity() * 608) == 547  # budget still exact


def test_every_output_unit_keeps_an_input():
    weights, masks, pruner = make(min_keep_fraction=0.0, output_name="fc3")
    observe_random(weights, pruner)
    pruner.scores["fc3"][:, 1] *= 1e-9
    pruner.prune_to(0.95)
    assert np.all(pruner.mask("fc3").any(axis=0))


def test_infeasible_budget_raises_without_changing_masks():
    weights, masks, pruner = make(min_keep_fraction=0.5)
    observe_random(weights, pruner)
    with pytest.raises(ValueError, match="safeguards"):
        pruner.prune_to(0.9)
    assert pruner.sparsity() == 0.0


# ------------------------------------------------------------- enforcement
def test_enforce_zeros_weights_grads_and_calls_set_mask():
    weights, masks, pruner = make()
    observe_random(weights, pruner)
    ids = {n: id(m.data) for n, m in masks.items()}
    opt = RecordingOptimizer()
    pruner.prune_to(0.7, optimizer=opt)
    for name, p in weights.items():
        mask = pruner.mask(name)
        assert np.all(p.data[~mask] == 0) and np.all(p.grad[~mask] == 0)
        assert id(masks[name].data) == ids[name]
    assert [c[0] for c in opt.calls] == list(weights.values())
    assert all(c[1].dtype == bool for c in opt.calls)
    pruner.check_integrity()


def test_optimizer_without_set_mask_is_rejected():
    weights, masks, pruner = make()
    observe_random(weights, pruner)
    with pytest.raises(TypeError, match="mask_state"):
        pruner.prune_to(0.5, optimizer=object())


def test_check_integrity_detects_external_edits():
    weights, masks, pruner = make()
    observe_random(weights, pruner)
    pruner.prune_to(0.5)
    dead = tuple(np.argwhere(~pruner.mask("fc2"))[0])
    weights["fc2"].data[dead] = 1.0
    with pytest.raises(RuntimeError, match="nonzero"):
        pruner.check_integrity()
    weights["fc2"].data[dead] = 0.0
    masks["fc2"].data[dead] = 1.0  # regrowth behind the pruner's back
    with pytest.raises(RuntimeError, match="outside"):
        pruner.check_integrity()


def test_constructor_validation():
    w, m = {"a": T(np.ones((2, 2)))}, {"a": T(np.ones((2, 3)))}
    with pytest.raises(ValueError, match="shape"):
        GlobalTaylorPruner(w, m)
    with pytest.raises(ValueError, match="keys"):
        GlobalTaylorPruner(w, {"b": T(np.ones((2, 2)))})
    with pytest.raises(ValueError, match="0 and 1"):
        GlobalTaylorPruner(w, {"a": T(np.full((2, 2), 0.5))})
    with pytest.raises(ValueError):
        GlobalTaylorPruner(w, {"a": T(np.ones((2, 2)))}, criterion="random")


def test_min_per_output_unit_keeps_k_inputs():
    weights, masks, pruner = make(min_keep_fraction=0.0, output_name="fc3",
                                  min_per_output_unit=3)
    observe_random(weights, pruner)
    pruner.scores["fc3"] *= 1e-9
    pruner.prune_to(0.95)
    assert np.all(pruner.mask("fc3").sum(axis=0) >= 3)


def test_layer_report_shape():
    weights, masks, pruner = make()
    observe_random(weights, pruner)
    pruner.prune_to(0.5)
    rep = pruner.layer_report()
    assert set(rep) == set(SHAPES)
    for name, r in rep.items():
        assert r["total"] == np.prod(SHAPES[name])
        assert r["sparsity"] == pytest.approx(1 - r["kept"] / r["total"])