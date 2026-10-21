
import numpy as np
import pytest

from pytrainer.autodiff.tensor import Tensor
from pytrainer.autodiff.losses import softmax_cross_entropy


def test_loss_with_integer_targets():
    logits = Tensor(
        [[2.0, 1.0, 0.1], [0.5, 1.5, 2.5]],
        requires_grad=True,
    )
    targets = np.array([0, 2])

    loss = softmax_cross_entropy(logits, targets)

    shifted = logits.data - logits.data.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(
        np.exp(shifted).sum(axis=1, keepdims=True)
    )
    expected = -np.mean(log_probs[np.arange(2), targets])

    np.testing.assert_allclose(loss.data, expected)
    assert loss.shape == ()


def test_integer_and_one_hot_targets_match():
    values = np.array([[1.0, 2.0, 0.0], [3.0, 1.0, 2.0]])
    indices = np.array([1, 0])
    one_hot = np.eye(3)[indices]

    loss_indices = softmax_cross_entropy(
        Tensor(values.copy()), indices
    )
    loss_one_hot = softmax_cross_entropy(
        Tensor(values.copy()), one_hot
    )

    np.testing.assert_allclose(loss_indices.data, loss_one_hot.data)


def test_gradient_matches_softmax_minus_targets():
    values = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 3.0]])
    targets = np.array([0, 2])

    logits = Tensor(values.copy(), requires_grad=True)
    loss = softmax_cross_entropy(logits, targets)
    loss.backward()

    shifted = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probs = exp_values / exp_values.sum(axis=1, keepdims=True)

    one_hot = np.eye(3)[targets]
    expected_grad = (probs - one_hot) / len(targets)

    np.testing.assert_allclose(
        logits.grad, expected_grad, rtol=1e-7, atol=1e-9
    )


@pytest.mark.parametrize(
    "values",
    [
        [[10000.0, 9999.0, 9998.0],
         [-10000.0, -9999.0, -9998.0]],
        [[1000.0, -1000.0, 0.0],
         [-1000.0, 1000.0, 0.0]],
    ],
)
def test_extreme_logits_are_stable(values):
    logits = Tensor(np.array(values), requires_grad=True)
    targets = np.array([0, 1])

    loss = softmax_cross_entropy(logits, targets)
    loss.backward()

    assert np.isfinite(loss.data)
    assert np.all(np.isfinite(logits.grad))


def test_batch_mean_normalization():
    values = np.array([[2.0, 0.0], [0.0, 2.0]])
    targets = np.array([0, 1])

    logits = Tensor(values.copy(), requires_grad=True)
    loss = softmax_cross_entropy(logits, targets)
    loss.backward()

    duplicated = Tensor(
        np.concatenate([values, values], axis=0),
        requires_grad=True,
    )
    duplicated_targets = np.concatenate([targets, targets])
    duplicated_loss = softmax_cross_entropy(
        duplicated, duplicated_targets
    )
    duplicated_loss.backward()

    # Duplicating the batch preserves the mean loss.
    np.testing.assert_allclose(
        duplicated_loss.data, loss.data
    )

    # Each example's gradient is halved because N doubles.
    np.testing.assert_allclose(
        duplicated.grad[:2], logits.grad / 2
    )
    np.testing.assert_allclose(
        duplicated.grad[2:], logits.grad / 2
    )



def test_finite_difference_gradient():
    values = np.array([[0.3, -0.2, 1.1], [1.2, 0.1, -0.7]])
    targets = np.array([2, 0])
    eps = 1e-6

    logits = Tensor(values.copy(), requires_grad=True)
    softmax_cross_entropy(logits, targets).backward()

    numerical = np.zeros_like(values)

    for idx in np.ndindex(values.shape):
        plus = values.copy()
        minus = values.copy()
        plus[idx] += eps
        minus[idx] -= eps

        numerical[idx] = (
            softmax_cross_entropy(Tensor(plus), targets).data
            - softmax_cross_entropy(Tensor(minus), targets).data
        ) / (2 * eps)

    np.testing.assert_allclose(
        logits.grad, numerical, rtol=1e-5, atol=1e-7
    )


def test_invalid_target_shapes_are_rejected():
    logits = Tensor(np.zeros((2, 3)))

    with pytest.raises(ValueError, match="targets must have shape"):
        softmax_cross_entropy(logits, np.zeros((2, 1)))


def test_out_of_range_class_index_is_rejected():
    logits = Tensor(np.zeros((2, 3)))

    with pytest.raises(ValueError, match="out of range"):
        softmax_cross_entropy(logits, np.array([0, 3]))
