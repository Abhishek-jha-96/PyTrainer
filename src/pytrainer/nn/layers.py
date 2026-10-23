"""Neural-network layers built on top of the autodiff engine."""

import numpy as np

from pytrainer.autodiff import ops
from pytrainer.autodiff.tensor import Tensor


# ----------------------------------------------------------------------
# Weight initializers
#
# Storage layout: weight.shape == (fan_in, fan_out) and the forward pass
# computes x @ W. (PyTorch stores Linear weights as (out, in); copying its
# indexing would swap fan_in and fan_out here.)
# ----------------------------------------------------------------------
def he_normal(shape, rng):
    """He (Kaiming) normal for ReLU layers: N(0, 2 / fan_in)."""
    fan_in, _ = shape
    return rng.normal(loc=0.0, scale=np.sqrt(2.0 / fan_in), size=shape)


def xavier_uniform(shape, rng):
    """Xavier (Glorot) uniform for linear layers: U(-a, a), a = sqrt(6 / (fan_in + fan_out))."""
    fan_in, fan_out = shape
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(low=-a, high=a, size=shape)


class Linear:
    """Fully connected layer computing ``Y = X @ (W * M) + b``.

    Shapes:
        x:           (N, in_features)
        weight:      (in_features, out_features)   i.e. (fan_in, fan_out)
        weight_mask: (in_features, out_features)   binary, non-trainable
        bias:        (out_features,)
        output:      (N, out_features)

    ``M`` is a constant, so ``dL/dW = M * dL/dW_effective``: masked
    connections get zero gradient and have no effect on the output. The
    mask starts dense (all ones), is written in place by the pruner, and is
    not returned by ``parameters()``, so the optimizer never updates it.

    Args:
        in_features, out_features: layer dimensions.
        rng: a seeded ``np.random.Generator``. Share one generator across
            all layers so a single seed reproduces the whole model.
        init: weight initializer, ``he_normal`` (ReLU hidden layers) or
            ``xavier_uniform`` (linear output layer). Biases start at zero.
    """

    def __init__(self, in_features, out_features, rng, init=he_normal):
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                "in_features and out_features must be positive; "
                f"got {in_features} and {out_features}"
            )
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Tensor(init((in_features, out_features), rng), requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self.weight_mask = Tensor(np.ones_like(self.weight.data))

    def __call__(self, x):
        # Wrap raw arrays: Tensor does not opt out of NumPy's operator
        # dispatch, so `ndarray @ Tensor` would bypass the engine.
        x = ops._tensor(x)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Linear expected input of shape (N, {self.in_features}); got {x.shape}"
            )
        effective_weight = ops.mul(self.weight, self.weight_mask)
        return ops.add(ops.matmul(x, effective_weight), self.bias)

    def parameters(self):
        # The mask is deliberately excluded: it must never reach the optimizer.
        return [self.weight, self.bias]
