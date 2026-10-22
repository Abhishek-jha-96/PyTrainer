import numpy as np
import pytest

from pytrainer.autodiff import ops
from pytrainer.autodiff.tensor import Tensor
from pytrainer.nn.optim import Adam

LR, B1, B2, EPS = 1e-3, 0.9, 0.999, 1e-8


def param(value):
    return Tensor(np.array(value, dtype=np.float64), requires_grad=True)


def reference_adam(theta, grads, lr=LR, b1=B1, b2=B2, eps=EPS):
    """Independent, loop-based Adam for comparison (returns final theta)."""
    theta = np.array(theta, dtype=np.float64)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta


# ----------------------------------------------------------------------
# Required tests
# ----------------------------------------------------------------------
@pytest.mark.parametrize("grad, direction", [(3.0, -1), (-3.0, +1)])
def test_scalar_moves_against_gradient(grad, direction):
    p = param(1.0)
    opt = Adam([p])
    p.grad = np.array(grad)
    opt.step()
    assert np.sign(p.data - 1.0) == direction


def test_first_update_matches_bias_corrected_formula():
    p = param([0.5, -2.0, 4.0])
    g = np.array([0.2, -7.0, 1e-3])
    opt = Adam([p])
    p.grad = g.copy()
    opt.step()

    # t = 1: m_hat = g, v_hat = g^2, so the step is lr * g / (|g| + eps).
    expected = np.array([0.5, -2.0, 4.0]) - LR * g / (np.abs(g) + EPS)
    np.testing.assert_allclose(p.data, expected, rtol=0, atol=1e-15)
    # And the step size is ~lr regardless of gradient magnitude.
    np.testing.assert_allclose(np.abs(p.data - [0.5, -2.0, 4.0]), LR, rtol=1e-4)


def test_parameters_with_different_shapes_match_reference():
    rng = np.random.default_rng(0)
    shapes = [(), (5,), (3, 4), (2, 3, 2)]
    init = [rng.standard_normal(s) for s in shapes]
    params = [param(x.copy()) for x in init]
    opt = Adam(params)

    grad_hist = [[] for _ in shapes]
    for _ in range(25):
        for k, (p, s) in enumerate(zip(params, shapes)):
            g = rng.standard_normal(s)
            grad_hist[k].append(g)
            p.grad = g.copy()
        opt.step()

    for k, (p, m, v, s) in enumerate(zip(params, opt.m, opt.v, shapes)):
        assert p.data.shape == m.shape == v.shape == s
        np.testing.assert_allclose(
            p.data, reference_adam(init[k], grad_hist[k]), rtol=1e-12, atol=1e-14
        )


def test_moments_persist_between_steps():
    p = param([1.0, 2.0])
    opt = Adam([p])
    m_obj, v_obj = opt.m[0], opt.v[0]

    g1 = np.array([1.0, -2.0])
    p.grad = g1.copy()
    opt.step()
    m1, v1 = opt.m[0].copy(), opt.v[0].copy()
    np.testing.assert_allclose(m1, (1 - B1) * g1)
    np.testing.assert_allclose(v1, (1 - B2) * g1**2)

    opt.zero_grad()  # clears grads only, not optimizer state
    g2 = np.array([0.5, 3.0])
    p.grad = g2.copy()
    opt.step()

    assert opt.m[0] is m_obj and opt.v[0] is v_obj  # same state arrays
    np.testing.assert_allclose(opt.m[0], B1 * m1 + (1 - B1) * g2)
    np.testing.assert_allclose(opt.v[0], B2 * v1 + (1 - B2) * g2**2)
    assert opt.t == 2


def test_zero_gradient_leaves_parameter_unchanged():
    start = np.array([[1.5, -0.5], [0.0, 3.0]])
    p = param(start.copy())
    opt = Adam([p])
    for _ in range(10):
        p.grad = np.zeros_like(p.data)
        opt.step()
    np.testing.assert_array_equal(p.data, start)


def test_repeated_updates_stay_finite():
    rng = np.random.default_rng(1)
    p = param(rng.standard_normal((8, 4)))
    opt = Adam([p])
    for i in range(2000):
        # Mix ordinary, tiny and large (but finite) gradient magnitudes.
        scale = [1.0, 1e-12, 1e6][i % 3]
        p.grad = rng.standard_normal(p.shape) * scale
        opt.step()
        assert np.all(np.isfinite(p.data))
        assert np.all(np.isfinite(opt.m[0])) and np.all(np.isfinite(opt.v[0]))


# ----------------------------------------------------------------------
# Implementation-detail tests
# ----------------------------------------------------------------------
def test_step_counter_is_shared_across_parameters():
    a, b = param(1.0), param([1.0, 2.0])
    opt = Adam([a, b])
    for _ in range(3):
        a.grad = np.array(1.0)
        b.grad = np.ones(2)
        opt.step()
    assert opt.t == 3
    # Both followed the same bias-correction schedule.
    np.testing.assert_allclose(a.data, reference_adam(1.0, [np.array(1.0)] * 3))
    np.testing.assert_allclose(b.data, reference_adam([1.0, 2.0], [np.ones(2)] * 3))


def test_update_is_in_place_and_preserves_identity():
    p = param([1.0, 2.0])
    tensor_id, array_id = id(p), id(p.data)
    opt = Adam([p])
    p.grad = np.array([1.0, 1.0])
    opt.step()
    assert opt.parameters[0] is p and id(p) == tensor_id
    assert id(p.data) == array_id  # same ndarray, mutated


def test_step_creates_no_graph_nodes():
    p = param([1.0, 2.0])
    opt = Adam([p])
    p.grad = np.array([1.0, 1.0])
    opt.step()
    assert p._parents == () and p._backward is None


def test_zero_grad_prevents_gradient_leakage():
    # loss = sum(w * x): dloss/dw = x. Without zero_grad, grads would add.
    w = param([0.0, 0.0])
    opt = Adam([w])
    x = Tensor([2.0, -1.0])
    for _ in range(3):
        opt.zero_grad()
        ops.sum(ops.mul(w, x)).backward()
        np.testing.assert_allclose(w.grad, [2.0, -1.0])
        opt.step()


def test_parameter_without_grad_is_skipped():
    a, b = param(1.0), param(2.0)
    opt = Adam([a, b])
    a.grad = np.array(1.0)  # b.grad stays None
    opt.step()
    assert a.data != 1.0
    assert b.data == 2.0 and opt.m[1] == 0.0 and opt.v[1] == 0.0


def test_grad_shape_mismatch_raises():
    p = param([1.0, 2.0])
    opt = Adam([p])
    p.grad = np.ones(3)
    with pytest.raises(ValueError):
        opt.step()


def test_non_finite_gradient_raises():
    p = param([1.0, 2.0])
    opt = Adam([p])
    p.grad = np.array([1.0, np.nan])
    with pytest.raises(FloatingPointError):
        opt.step()


@pytest.mark.parametrize(
    "kwargs",
    [{"lr": 0.0}, {"beta1": 1.0}, {"beta2": -0.1}, {"eps": 0.0}],
)
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        Adam([param(1.0)], **kwargs)


def test_invalid_parameter_lists_raise():
    p = param(1.0)
    with pytest.raises(ValueError):
        Adam([])
    with pytest.raises(ValueError):
        Adam([p, p])
    with pytest.raises(ValueError):
        Adam([Tensor(1.0)])  # requires_grad=False
    with pytest.raises(TypeError):
        Adam([np.array(1.0)])


def test_defaults_match_pytorch():
    opt = Adam([param(1.0)])
    assert (opt.lr, opt.beta1, opt.beta2, opt.eps) == (1e-3, 0.9, 0.999, 1e-8)


def test_minimizes_quadratic_with_autodiff_gradients():
    # loss = sum((w - target)^2) using engine ops; Adam should reach target.
    target = np.array([3.0, -1.0, 0.5])
    w = param(np.zeros(3))
    opt = Adam([w], lr=0.05)
    neg_target = Tensor(-target)
    for _ in range(1000):
        opt.zero_grad()
        diff = ops.add(w, neg_target)
        ops.sum(ops.mul(diff, diff)).backward()
        opt.step()
    np.testing.assert_allclose(w.data, target, atol=1e-3)