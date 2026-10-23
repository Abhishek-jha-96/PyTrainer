"""Tests for saliency scores, sparsity schedule, mask selection and MaskedAdam."""

import numpy as np
import pytest

from pytrainer.autodiff.tensor import Tensor
from pytrainer.nn.optim import Adam, MaskedAdam
from pytrainer.nn.pruning import (
    Pruner, SaliencyTracker, instantaneous_saliency, pruning_progress,
    select_masks, sparsity_schedule,
)


def param(x):
    return Tensor(np.array(x, dtype=np.float64), requires_grad=True)


# ----------------------------------------------------------------------
# Saliency
# ----------------------------------------------------------------------
def test_instantaneous_scores():
    w, g = np.array([2.0, -3.0, 0.0]), np.array([-0.5, 0.1, 9.0])
    np.testing.assert_allclose(instantaneous_saliency(w, g, "taylor_abs"), [1.0, 0.3, 0.0])
    np.testing.assert_allclose(instantaneous_saliency(w, g, "taylor_sq"), [1.0, 0.09, 0.0])
    np.testing.assert_allclose(instantaneous_saliency(w, g, "magnitude"), [2.0, 3.0, 0.0])
    with pytest.raises(ValueError):
        instantaneous_saliency(w, g, "random")


def test_first_bias_corrected_score_equals_instantaneous():
    w = param([[2.0, -1.0]])
    w.grad = np.array([[0.5, 4.0]])
    t = SaliencyTracker([("w", w)], "taylor_abs", decay=0.9)
    t.update()
    np.testing.assert_allclose(t.scores()["w"], [[1.0, 4.0]])  # no shrink toward 0


def test_ema_is_bias_corrected_average():
    w = param([1.0])
    t = SaliencyTracker([("w", w)], "taylor_abs", decay=0.5)
    for g in (1.0, 3.0):
        w.grad = np.array([g])
        t.update()
    # raw ema = 0.5*(0.5*1) + 0.5*3 = 1.75 ; correction 1 - 0.25 = 0.75
    np.testing.assert_allclose(t.scores()["w"], [1.75 / 0.75])


def test_abs_is_taken_per_batch_not_after_averaging():
    # Opposite gradients cancel in the mean but not in the per-batch score.
    w = param([1.0])
    t = SaliencyTracker([("w", w)], "taylor_abs", decay=0.5)
    for g in (2.0, -2.0):
        w.grad = np.array([g])
        t.update()
    np.testing.assert_allclose(t.scores()["w"], [2.0])


def test_tracker_requires_gradients():
    w = param([1.0])
    t = SaliencyTracker([("w", w)], "taylor_abs")
    with pytest.raises(RuntimeError):
        t.update()
    with pytest.raises(RuntimeError):
        SaliencyTracker([("w", w)]).scores()


# ----------------------------------------------------------------------
# Schedule: s(t) = s_f [1 - (1 - u)^3],  u = clip((t - t0)/(t1 - t0), 0, 1)
# ----------------------------------------------------------------------
def test_progress_is_clipped():
    assert pruning_progress(0, 10, 80) == 0.0
    assert pruning_progress(45, 10, 80) == 0.5
    assert pruning_progress(200, 10, 80) == 1.0


def test_schedule_phases_and_formula():
    t0, t1, sf = 10, 80, 0.9
    s = lambda t: sparsity_schedule(t, t0, t1, sf)
    assert all(s(t) == 0.0 for t in range(0, t0 + 1))           # warm-up
    assert s(t1) == pytest.approx(sf)
    assert all(s(t) == pytest.approx(sf) for t in range(t1, 101))  # fine-tune
    for t in (11, 30, 45, 79):
        u = (t - t0) / (t1 - t0)
        assert s(t) == pytest.approx(sf * (1 - (1 - u) ** 3))
    vals = [s(t) for t in range(0, 101)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))           # monotone


def test_schedule_is_front_loaded():
    t0, t1, sf = 10, 80, 0.9
    assert sparsity_schedule(45, t0, t1, sf) == pytest.approx(0.875 * sf)  # u = 0.5
    first = sparsity_schedule(11, t0, t1, sf) - sparsity_schedule(10, t0, t1, sf)
    last = sparsity_schedule(80, t0, t1, sf) - sparsity_schedule(79, t0, t1, sf)
    assert first > 100 * last  # pruning rate tapers toward the target


def test_schedule_validates_arguments():
    with pytest.raises(ValueError):
        sparsity_schedule(5, 10, 10)
    with pytest.raises(ValueError):
        sparsity_schedule(5, 10, 80, 1.0)


# ----------------------------------------------------------------------
# Mask selection
# ----------------------------------------------------------------------
def all_alive(scores):
    return {n: np.ones(s.shape, dtype=bool) for n, s in scores.items()}


def test_keeps_exact_budget_of_global_top_scores():
    rng = np.random.default_rng(0)
    scores = {"a": rng.random((4, 8)), "b": rng.random((8, 4))}
    masks = select_masks(scores, all_alive(scores), 0.75, min_keep_fraction=0.0)
    assert sum(m.sum() for m in masks.values()) == 16
    flat = np.concatenate([scores[n].ravel() for n in scores])
    kept = np.concatenate([scores[n][masks[n]] for n in scores])
    np.testing.assert_allclose(np.sort(kept), np.sort(flat)[-16:])


def test_per_layer_floor_protects_low_scoring_layer():
    scores = {"big": np.ones((10, 10)), "tiny": np.full((2, 5), 1e-9)}
    masks = select_masks(scores, all_alive(scores), 0.9, min_keep_fraction=0.2)
    assert masks["tiny"].sum() >= 2  # ceil(0.2 * 10)
    no_floor = select_masks(scores, all_alive(scores), 0.9, min_keep_fraction=0.0)
    assert no_floor["tiny"].sum() == 0


def test_every_output_unit_keeps_an_input():
    scores = {"hid": np.ones((4, 4)), "out": np.array([[1e-9, 1.0], [1e-9, 1.0]])}
    masks = select_masks(scores, all_alive(scores), 0.9, min_keep_fraction=0.0,
                         output_layer="out", min_per_output_unit=1)
    assert np.all(masks["out"].any(axis=0))


def test_masks_are_monotone():
    rng = np.random.default_rng(1)
    scores = {"a": rng.random((5, 5))}
    m1 = select_masks(scores, all_alive(scores), 0.5, min_keep_fraction=0.0)
    boosted = {"a": np.where(m1["a"], 0.0, 100.0)}  # pruned weights now score highest
    m2 = select_masks(boosted, m1, 0.6, min_keep_fraction=0.0)
    assert not np.any(m2["a"] & ~m1["a"])


# ----------------------------------------------------------------------
# MaskedAdam
# ----------------------------------------------------------------------
def test_pruned_weights_stay_exactly_zero_under_momentum():
    p = param([1.0, 2.0, 3.0])
    opt = MaskedAdam([p])
    for _ in range(5):  # build up momentum on every entry
        p.grad = np.array([1.0, 1.0, 1.0])
        opt.step()
    opt.set_mask(p, np.array([True, False, True]))
    assert p.data[1] == 0.0 and opt.m[0][1] == 0.0 and opt.v[0][1] == 0.0
    for _ in range(50):
        p.grad = np.array([1.0, 5.0, -1.0])
        opt.step()
        assert p.data[1] == 0.0


def test_plain_adam_would_move_a_weight_with_momentum():
    # Documents why the moments must be reset: zero grad alone is not enough.
    p = param([1.0])
    opt = Adam([p])
    for _ in range(5):
        p.grad = np.array([1.0]); opt.step()
    p.data[0] = 0.0
    p.grad = np.array([0.0]); opt.step()
    assert p.data[0] != 0.0


def test_unmasked_entries_follow_plain_adam_exactly():
    rng = np.random.default_rng(2)
    init = rng.standard_normal(4)
    a, b = param(init.copy()), param(init.copy())
    bias_a, bias_b = param([0.5]), param([0.5])
    plain, masked = Adam([a, bias_a]), MaskedAdam([b, bias_b])
    masked.set_mask(b, np.array([True, True, False, True]))
    a.data[2] = 0.0; plain.m[0][2] = plain.v[0][2] = 0.0
    for _ in range(20):
        g, gb = rng.standard_normal(4), rng.standard_normal(1)
        a.grad, b.grad = np.where([1, 1, 0, 1], g, 0.0), g.copy()
        bias_a.grad, bias_b.grad = gb.copy(), gb.copy()
        plain.step(); masked.step()
    np.testing.assert_allclose(b.data, a.data, rtol=1e-12)
    np.testing.assert_allclose(bias_b.data, bias_a.data, rtol=1e-12)


def test_set_mask_validates():
    p, q = param([1.0, 2.0]), param([1.0])
    opt = MaskedAdam([p])
    with pytest.raises(ValueError):
        opt.set_mask(q, np.array([True]))
    with pytest.raises(ValueError):
        opt.set_mask(p, np.array([True]))


# ----------------------------------------------------------------------
# Pruner
# ----------------------------------------------------------------------
def make_layers(seed=3):
    from pytrainer.nn.layers import Linear
    rng = np.random.default_rng(seed)
    return Linear(4, 6, rng), Linear(6, 2, rng, init="xavier_uniform"), rng


def test_pruner_installs_masks_in_layers_and_optimizer():
    l1, l2, rng = make_layers()
    opt = MaskedAdam(l1.parameters() + l2.parameters())
    pr = Pruner([("l1", l1), ("l2", l2)], opt, "taylor_abs",
                min_keep_fraction=0.0, output_layer="l2.weight")
    x = Tensor(rng.standard_normal((8, 4)))
    from pytrainer.autodiff import ops as _ops
    _ops.sum(l2(_ops.relu(l1(x)))).backward()
    pr.update_scores()
    assert pr.prune_to(0.75) == pytest.approx(0.75)
    for key, layer in (("l1.weight", l1), ("l2.weight", l2)):
        m = pr.masks[key]
        np.testing.assert_array_equal(layer.weight_mask.data.astype(bool), m)  # forward graph
        assert np.all(layer.weight.data[~m] == 0.0)                            # raw weights
    pr.check_integrity()
    assert pr.layer_report()["l2.weight"]["units_without_inputs"] == 0


def test_masked_connections_get_zero_gradient_after_pruning():
    l1, l2, rng = make_layers()
    opt = MaskedAdam(l1.parameters() + l2.parameters())
    pr = Pruner([("l1", l1), ("l2", l2)], opt, "magnitude", min_keep_fraction=0.0)
    pr.update_scores()
    pr.prune_to(0.5)
    from pytrainer.autodiff import ops as _ops
    for p in opt.parameters:
        p.grad = None
    _ops.sum(l2(_ops.relu(l1(Tensor(rng.standard_normal((8, 4))))))).backward()
    assert np.all(l1.weight.grad[~pr.masks["l1.weight"]] == 0.0)
    assert np.all(l2.weight.grad[~pr.masks["l2.weight"]] == 0.0)


def test_integrity_check_catches_divergence():
    l1, l2, _ = make_layers()
    opt = MaskedAdam(l1.parameters() + l2.parameters())
    pr = Pruner([("l1", l1), ("l2", l2)], opt, "magnitude", min_keep_fraction=0.0)
    pr.update_scores()
    pr.prune_to(0.5)
    m = pr.masks["l1.weight"]
    l1.weight.data[~m] = 1.0                 # raw weight revived behind the mask
    with pytest.raises(AssertionError):
        pr.check_integrity()
    l1.weight.data[~m] = 0.0
    l1.set_weight_mask(np.ones_like(m))      # layer mask changed without the optimizer
    with pytest.raises(AssertionError):
        pr.check_integrity()


# ----------------------------------------------------------------------
# End-to-end schedule (short)
# ----------------------------------------------------------------------
from pytrainer.experiments.data import prepare_two_moons
from pytrainer.experiments.prune_mlp import PruneConfig, run, train_with_pruning
from pytrainer.experiments.train_mlp import TrainConfig, train

SHORT = dict(start_epoch=3, end_epoch=8, finetune_epochs=3, log_every=0)


@pytest.fixture(scope="module")
def moons():
    return prepare_two_moons(seed=42)


@pytest.fixture(scope="module")
def short_run(moons):
    return train_with_pruning(PruneConfig(**SHORT), moons)


def test_phases_follow_schedule(short_run):
    h = short_run["history"]
    assert h["epoch"] == list(range(0, 12))
    assert h["phase"] == ["init"] + ["warmup"] * 3 + ["prune"] * 5 + ["finetune"] * 3


def test_measured_sparsity_tracks_target_within_one_weight(short_run):
    h = short_run["history"]
    for target, actual in zip(h["target_sparsity"], h["sparsity"]):
        assert abs(actual - target) <= 1 / 608 + 1e-12


def test_warmup_matches_dense_training(short_run, moons):
    dense = train(TrainConfig(seed=42, max_epochs=3, patience=0, log_every=0), data=moons).history
    h = short_run["history"]
    for k in ("train_loss", "val_loss", "train_accuracy", "val_accuracy"):
        np.testing.assert_allclose(h[k][:4], dense[k][:4], rtol=1e-12)


def test_selection_only_at_target_sparsity(short_run):
    h = short_run["history"]
    best = short_run["best_epoch"]
    assert best >= 8
    i = h["epoch"].index(best)
    assert h["val_loss"][i] == min(h["val_loss"][8:])


def test_budget_counts_only_weights(short_run):
    pr = short_run["pruner"]
    assert sum(m.size for m in pr.masks.values()) == 608  # 64 + 512 + 32; biases excluded
    assert int(sum(m.sum() for m in pr.masks.values())) == round(0.1 * 608)
    for n, p in named_parameters_of(short_run["model"]):
        if n.endswith(".bias"):
            assert np.count_nonzero(p.data) > 0  # biases are never pruned


def named_parameters_of(model):
    from pytrainer.experiments.train_mlp import named_parameters
    return named_parameters(model)


def test_global_ranking_allocates_unevenly(short_run):
    rep = short_run["pruner"].layer_report()
    keep = [1 - rep[n]["sparsity"] for n in ("fc1.weight", "fc2.weight", "fc3.weight")]
    assert max(keep) - min(keep) > 0.1  # not a uniform 10% per layer


def test_masks_hold_and_run_is_reproducible(short_run, moons):
    pr = short_run["pruner"]
    pr.check_integrity()
    for n, w in pr.named_weights:
        assert np.all(w.data[~pr.masks[n]] == 0.0)
    again = train_with_pruning(PruneConfig(**SHORT), moons)
    assert again["history"] == short_run["history"]


def test_run_writes_artifacts(tmp_path):
    summary, _ = run(PruneConfig(**SHORT), tmp_path)
    for f in ("pruning_history.csv", "pruning_loss_curve.png",
              "pruning_accuracy_curve.png", "sparsity_schedule.png", "pruning_summary.json"):
        assert (tmp_path / f).stat().st_size > 0
    assert summary["budget"]["eligible_weights"] == 608
    assert summary["biases_dense"] == 50


def test_invalid_schedule_config():
    with pytest.raises(ValueError):
        train_with_pruning(PruneConfig(start_epoch=10, end_epoch=10), None)