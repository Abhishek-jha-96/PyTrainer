import numpy as np
import pytest

from pytrainer.experiments.data import (
    Standardizer,
    generate_two_moons,
    iterate_minibatches,
    prepare_two_moons,
    stratified_split,
)

SEED = 42


@pytest.fixture(scope="module")
def data():
    return prepare_two_moons(seed=SEED)


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("backend", ["sklearn", "numpy"])
def test_generation_shapes_dtypes_and_labels(backend):
    X, y = generate_two_moons(2000, 0.2, random_state=SEED, backend=backend)
    assert X.shape == (2000, 2) and X.dtype == np.float64
    assert y.shape == (2000,) and y.dtype == np.int64
    assert set(np.unique(y)) == {0, 1}
    assert np.all(np.isfinite(X))


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        generate_two_moons(backend="torch")


# ----------------------------------------------------------------------
# Split
# ----------------------------------------------------------------------
def test_split_sizes_and_shapes(data):
    assert data.X_train.shape == (1400, 2) and data.y_train.shape == (1400,)
    assert data.X_val.shape == (300, 2) and data.y_val.shape == (300,)
    assert data.X_test.shape == (300, 2) and data.y_test.shape == (300,)
    for y in (data.y_train, data.y_val, data.y_test):
        assert y.dtype == np.int64 and set(np.unique(y)) <= {0, 1}


def test_splits_are_disjoint_and_cover_everything(data):
    tr, va, te = (set(data.indices[k]) for k in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(range(len(data.y_raw)))


def test_splits_are_class_balanced(data):
    overall = data.y_raw.mean()
    for y in (data.y_train, data.y_val, data.y_test):
        assert abs(y.mean() - overall) < 0.01


def test_split_is_stratified_with_unbalanced_classes():
    y = np.array([0] * 900 + [1] * 100)
    idx = stratified_split(y, np.random.default_rng(0))
    for name in ("train", "val", "test"):
        np.testing.assert_allclose(y[idx[name]].mean(), 0.1, atol=0.01)


def test_labels_align_with_raw_data(data):
    np.testing.assert_array_equal(data.y_train, data.y_raw[data.indices["train"]])
    np.testing.assert_array_equal(data.y_test, data.y_raw[data.indices["test"]])


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def test_same_seed_reproduces_everything(data):
    again = prepare_two_moons(seed=SEED)
    for name in ("train", "val", "test"):
        np.testing.assert_array_equal(data.indices[name], again.indices[name])
    np.testing.assert_array_equal(data.X_train, again.X_train)
    np.testing.assert_array_equal(data.X_test, again.X_test)


def test_different_seed_changes_split(data):
    other = prepare_two_moons(seed=SEED + 1)
    assert not np.array_equal(data.indices["train"], other.indices["train"])


# ----------------------------------------------------------------------
# Standardization and leakage
# ----------------------------------------------------------------------
def test_scaler_is_fitted_on_train_only(data):
    X_train_raw = data.X_raw[data.indices["train"]]
    np.testing.assert_allclose(data.scaler.mean, X_train_raw.mean(axis=0))
    np.testing.assert_allclose(data.scaler.std, X_train_raw.std(axis=0))
    # Statistics of the full dataset differ, so using them would be detectable.
    assert not np.allclose(data.scaler.mean, data.X_raw.mean(axis=0))


def test_train_is_standardized_and_heldout_uses_train_stats(data):
    np.testing.assert_allclose(data.X_train.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(data.X_train.std(axis=0), 1.0, atol=1e-12)
    for split in ("val", "test"):
        raw = data.X_raw[data.indices[split]]
        expected = (raw - data.scaler.mean) / data.scaler.std
        np.testing.assert_allclose(getattr(data, f"X_{split}"), expected)
        # Held-out data is not re-centred on its own statistics.
        assert not np.allclose(getattr(data, f"X_{split}").mean(axis=0), 0.0, atol=1e-12)


def test_test_set_cannot_influence_preprocessing():
    # Corrupt the test rows of the raw data: train-fitted stats must not move.
    base = prepare_two_moons(seed=SEED)
    X = base.X_raw.copy()
    X[base.indices["test"]] *= 1000.0
    refit = Standardizer().fit(X[base.indices["train"]])
    np.testing.assert_array_equal(refit.mean, base.scaler.mean)
    np.testing.assert_array_equal(refit.std, base.scaler.std)


def test_standardized_values_are_finite(data):
    for X in (data.X_train, data.X_val, data.X_test):
        assert np.all(np.isfinite(X))


def test_constant_feature_does_not_divide_by_zero():
    X = np.column_stack([np.arange(5.0), np.full(5, 3.0)])
    out = Standardizer().fit(X).transform(X)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[:, 1], 0.0)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        Standardizer().transform(np.ones((2, 2)))


# ----------------------------------------------------------------------
# Minibatches
# ----------------------------------------------------------------------
def test_minibatch_sizes_include_final_partial_batch(data):
    rng = np.random.default_rng(0)
    sizes = [len(xb) for xb, _ in iterate_minibatches(data.X_train, data.y_train, 64, rng)]
    assert sizes == [64] * 21 + [56]  # 1400 = 21 * 64 + 56


def test_one_epoch_covers_every_sample_once(data):
    rng = np.random.default_rng(0)
    seen = np.concatenate([xb for xb, _ in iterate_minibatches(data.X_train, data.y_train, 64, rng)])
    order = np.lexsort(seen.T)
    np.testing.assert_array_equal(seen[order], data.X_train[np.lexsort(data.X_train.T)])


def test_batches_keep_inputs_and_labels_paired(data):
    rng = np.random.default_rng(0)
    lookup = {tuple(x): label for x, label in zip(data.X_train, data.y_train)}
    for xb, yb in iterate_minibatches(data.X_train, data.y_train, 64, rng):
        assert xb.shape[1] == 2 and xb.shape[0] == yb.shape[0]
        assert all(lookup[tuple(x)] == label for x, label in zip(xb, yb))


def test_shuffle_differs_per_epoch_but_is_reproducible(data):
    def epoch_orders(seed, epochs=3):
        rng = np.random.default_rng(seed)
        return [
            np.concatenate([yb for _, yb in iterate_minibatches(
                np.arange(1400)[:, None], np.arange(1400), 64, rng)])
            for _ in range(epochs)
        ]
    a, b = epoch_orders(7), epoch_orders(7)
    for ea, eb in zip(a, b):
        np.testing.assert_array_equal(ea, eb)
    assert not np.array_equal(a[0], a[1])


def test_unshuffled_iteration_preserves_order(data):
    ys = np.concatenate([yb for _, yb in iterate_minibatches(data.X_val, data.y_val, 64, shuffle=False)])
    np.testing.assert_array_equal(ys, data.y_val)


def test_shuffle_without_rng_raises(data):
    with pytest.raises(ValueError):
        next(iterate_minibatches(data.X_train, data.y_train, 64))


# ======================================================================
# Section 2: training loop, evaluation and reporting (Subtasks 5 and 7)
# ======================================================================
import csv
import json

from pytrainer.autodiff.losses import softmax_cross_entropy
from pytrainer.autodiff.tensor import Tensor
from pytrainer.experiments.metrics import (
    HISTORY_FIELDS,
    accuracy,
    evaluate,
    load_history_csv,
    save_history_csv,
)
from pytrainer.experiments.train_mlp import (
    TrainConfig,
    check_finite_gradients,
    make_generators,
    named_parameters,
    report,
    train,
    train_one_epoch,
)
from pytrainer.nn.models import MLP
from pytrainer.nn.optim import Adam


def quick_config(**overrides):
    cfg = dict(seed=SEED, max_epochs=5, patience=0, log_every=0)
    cfg.update(overrides)
    return TrainConfig(**cfg)


@pytest.fixture(scope="module")
def model():
    init_rng, _ = make_generators(SEED)
    return MLP(init_rng)


@pytest.fixture(scope="module")
def full_run(data):
    """One real training run with the default stopping rule, shared by tests."""
    return train(TrainConfig(seed=SEED, log_every=0), data=data)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def test_accuracy_formula():
    logits = np.array([[2.0, 1.0], [0.0, 3.0], [5.0, -1.0], [0.1, 0.2]])
    assert accuracy(logits, np.array([0, 1, 1, 1])) == 0.75


def test_evaluate_loss_is_sample_weighted_and_batch_size_invariant(model, data):
    full = softmax_cross_entropy(model(Tensor(data.X_val)), data.y_val).item()
    for bs in (7, 64, 256, 300, 1000):
        m = evaluate(model, data.X_val, data.y_val, batch_size=bs)
        np.testing.assert_allclose(m["loss"], full, rtol=1e-12)
        assert m["n"] == 300


def test_evaluate_does_not_touch_params_or_grads(model, data):
    params = model.parameters()
    before = [p.data.copy() for p in params]
    for p in params:
        p.grad = None
    evaluate(model, data.X_train, data.y_train)
    for p, b in zip(params, before):
        np.testing.assert_array_equal(p.data, b)
        assert p.grad is None


# ----------------------------------------------------------------------
# Gradients and fail-fast checks
# ----------------------------------------------------------------------
def test_end_to_end_gradients_match_finite_differences(data):
    init_rng, _ = make_generators(SEED)
    m = MLP(init_rng)
    xb, yb = data.X_train[:32], data.y_train[:32]
    for p in m.parameters():
        p.grad = None
    softmax_cross_entropy(m(Tensor(xb)), yb).backward()

    f = lambda: softmax_cross_entropy(m(Tensor(xb)), yb).item()
    rng, eps = np.random.default_rng(0), 1e-6
    for name, p in named_parameters(m):
        for _ in range(5):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p.data[i]
            p.data[i] = old + eps; fp = f()
            p.data[i] = old - eps; fm = f()
            p.data[i] = old
            np.testing.assert_allclose(p.grad[i], (fp - fm) / (2 * eps), atol=1e-7, err_msg=name)


def test_check_finite_gradients_fails_fast():
    p = Tensor(np.ones(2), requires_grad=True)
    p.grad = np.array([1.0, np.inf])
    with pytest.raises(FloatingPointError, match="w"):
        check_finite_gradients([("w", p)])
    p.grad = None
    with pytest.raises(RuntimeError, match="no gradient"):
        check_finite_gradients([("w", p)])


def test_training_stops_on_non_finite_loss(data):
    init_rng, batch_rng = make_generators(SEED)
    m = MLP(init_rng)
    m.fc3.weight.data[0, 0] = np.nan
    opt = Adam(m.parameters())
    with pytest.raises(FloatingPointError):
        train_one_epoch(m, opt, data.X_train, data.y_train, 64, batch_rng, named_parameters(m))


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def test_one_epoch_takes_one_step_per_minibatch(data):
    init_rng, batch_rng = make_generators(SEED)
    m = MLP(init_rng)
    opt = Adam(m.parameters())
    steps = train_one_epoch(m, opt, data.X_train, data.y_train, 64, batch_rng, named_parameters(m))
    assert steps == 22 and opt.t == 22


def test_seed_streams_are_independent():
    init_a, batch_a = make_generators(SEED)
    split_rng = np.random.default_rng(np.random.SeedSequence(SEED).spawn(1)[0])
    draws = [g.random(4) for g in (init_a, batch_a, split_rng)]
    assert not np.allclose(draws[0], draws[1])
    assert not np.allclose(draws[0], draws[2])


def test_history_fields_baseline_and_finiteness(data):
    r = train(quick_config(), data=data)
    assert tuple(r.history) == HISTORY_FIELDS
    assert r.history["epoch"] == [0, 1, 2, 3, 4, 5]  # epoch 0 = untrained baseline
    for key in HISTORY_FIELDS[1:]:
        assert len(r.history[key]) == 6
        assert np.all(np.isfinite(r.history[key]))
    accs = r.history["train_accuracy"] + r.history["val_accuracy"]
    assert all(0.0 <= a <= 1.0 for a in accs)


def test_baseline_row_is_the_untrained_model(data):
    r = train(quick_config(max_epochs=1), data=data)
    init_rng, _ = make_generators(SEED)
    fresh = evaluate(MLP(init_rng), data.X_val, data.y_val)
    np.testing.assert_allclose(r.history["val_loss"][0], fresh["loss"])


def test_model_holds_selected_epoch_weights(data):
    r = train(quick_config(max_epochs=8), data=data)
    i = r.history["epoch"].index(r.best_epoch)
    assert r.history["val_loss"][i] == min(r.history["val_loss"])
    np.testing.assert_allclose(
        evaluate(r.model, data.X_val, data.y_val)["loss"], r.history["val_loss"][i]
    )
    np.testing.assert_allclose(
        evaluate(r.model, data.X_train, data.y_train)["accuracy"], r.history["train_accuracy"][i]
    )


def test_early_stopping_respects_patience(full_run):
    r = full_run
    if r.stopped_early:
        assert r.epochs_run - r.best_epoch == TrainConfig.patience
    assert len(r.history["epoch"]) == r.epochs_run + 1


def test_training_is_reproducible(data):
    h1 = train(quick_config(max_epochs=3), data=data).history
    h2 = train(quick_config(max_epochs=3), data=data).history
    assert h1 == h2


def test_training_never_reads_the_test_split(data):
    import dataclasses
    poisoned = dataclasses.replace(
        data, X_test=np.full_like(data.X_test, np.nan), y_test=np.full_like(data.y_test, -1)
    )
    h_clean = train(quick_config(max_epochs=2), data=data).history
    h_poisoned = train(quick_config(max_epochs=2), data=poisoned).history
    assert h_clean == h_poisoned


# ----------------------------------------------------------------------
# Learning outcome (real run)
# ----------------------------------------------------------------------
def test_full_run_reduces_loss_and_beats_target(full_run):
    h = full_run.history
    i = h["epoch"].index(full_run.best_epoch)
    assert h["train_loss"][-1] < 0.5 * h["train_loss"][0]
    assert h["val_loss"][i] < 0.5 * h["val_loss"][0]
    assert h["val_accuracy"][i] > 0.85
    assert h["val_accuracy"][i] > h["val_accuracy"][0]


# ----------------------------------------------------------------------
# Reporting outputs
# ----------------------------------------------------------------------
def test_history_csv_round_trip(tmp_path, data):
    h = train(quick_config(max_epochs=3), data=data).history
    path = tmp_path / "history.csv"
    save_history_csv(h, path)
    with open(path, newline="") as fh:
        assert tuple(next(csv.reader(fh))) == HISTORY_FIELDS
    assert load_history_csv(path) == h  # repr() keeps floats exact


def test_report_writes_all_artifacts(tmp_path, data):
    cfg = quick_config(max_epochs=3)
    r = train(cfg, data=data)
    summary = report(r, cfg, tmp_path)
    for name in ("history.csv", "loss_curve.png", "accuracy_curve.png", "summary.json"):
        assert (tmp_path / name).stat().st_size > 0
    with open(tmp_path / "loss_curve.png", "rb") as fh:
        assert fh.read(8) == b"\x89PNG\r\n\x1a\n"
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["selected_epoch"] == r.best_epoch
    assert saved["test"]["n"] == 300
    assert np.isfinite(saved["test"]["loss"]) and summary == saved