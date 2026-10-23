"""Neural-network layers built on top of the autodiff engine."""

import numpy as np

from pytrainer.autodiff import ops
from pytrainer.autodiff.tensor import Tensor


# ----------------------------------------------------------------------
# Weight initializers
#
# Storage layout: weight.shape == (fan_in, fan_out) and the forward pass
# computes x @ W. So fan_in is shape[0] and fan_out is shape[1].
# (PyTorch stores Linear weights as (out, in), where fan_in is shape[1];
# copying its indexing would swap fan_in and fan_out here.)
# ----------------------------------------------------------------------
def _fans(shape):
    if len(shape) != 2:
        raise ValueError(f"expected a 2-D weight shape, got {shape}")
    fan_in, fan_out = shape
    return fan_in, fan_out


def he_normal(shape, rng):
    """He (Kaiming) normal for ReLU layers: N(0, 2 / fan_in)."""
    fan_in, _ = _fans(shape)
    std = np.sqrt(2.0 / fan_in)
    return rng.normal(loc=0.0, scale=std, size=shape)


def xavier_uniform(shape, rng):
    """Xavier (Glorot) uniform for linear layers: U(-a, a), a = sqrt(6 / (fan_in + fan_out))."""
    fan_in, fan_out = _fans(shape)
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(low=-a, high=a, size=shape)


INITIALIZERS = {
    "he_normal": he_normal,
    "xavier_uniform": xavier_uniform,
}


class Linear:
    """Fully connected layer computing ``Y = X @ (W * M) + b``.

    Shapes:
        x:           (N, in_features)
        weight:      (in_features, out_features)   i.e. (fan_in, fan_out)
        weight_mask: (in_features, out_features)   binary, non-trainable
        bias:        (out_features,)
        output:      (N, out_features)

    The forward pass uses ``W_effective = W * M``. Because ``M`` is a
    constant (``requires_grad=False``), the engine's ``mul`` backward gives
    ``dL/dW = M * dL/dW_effective``: masked connections receive zero
    gradient and have no effect on the output, whatever value the raw
    weight holds. The mask starts as all ones (dense) and is never
    returned by ``parameters()``, so the optimizer cannot update it.

    Args:
        in_features, out_features: layer dimensions.
        rng: a seeded ``np.random.Generator``. Share one generator across
            all layers so a single seed reproduces the whole model.
        init: ``"he_normal"`` (ReLU hidden layers) or ``"xavier_uniform"``
            (linear output layer). Biases are always initialized to zero.
    """

    def __init__(self, in_features, out_features, rng, init="he_normal"):
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                "in_features and out_features must be positive; "
                f"got {in_features} and {out_features}"
            )
        if init not in INITIALIZERS:
            raise ValueError(
                f"unknown init {init!r}; choose from {sorted(INITIALIZERS)}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.init = init

        weight_data = INITIALIZERS[init]((in_features, out_features), rng)
        if not np.all(np.isfinite(weight_data)):
            raise FloatingPointError(f"{init} produced non-finite weights")

        self.weight = Tensor(weight_data, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)

        # Fixed, non-trainable binary mask (all ones = dense layer).
        self.weight_mask = Tensor(np.ones_like(self.weight.data), requires_grad=False)

    def set_weight_mask(self, mask):
        """Replace the mask in place (the mask Tensor keeps its identity)."""
        mask = np.asarray(mask)
        if mask.shape != self.weight.data.shape:
            raise ValueError(
                f"mask shape {mask.shape} != weight shape {self.weight.data.shape}"
            )
        values = mask.astype(np.float64)
        if not np.all((values == 0.0) | (values == 1.0)):
            raise ValueError("weight mask must be binary (0/1 or bool)")
        self.weight_mask.data[...] = values

    @property
    def active_weights(self):
        """Number of connections the mask keeps."""
        return int(np.count_nonzero(self.weight_mask.data))

    def __call__(self, x):
        # Wrap raw arrays: Tensor does not opt out of NumPy's operator
        # dispatch, so `ndarray @ Tensor` would bypass the engine.
        if not isinstance(x, Tensor):
            x = Tensor(x)

        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Linear expected input of shape (N, {self.in_features}); "
                f"got {x.shape}"
            )

        if self.weight_mask.requires_grad:
            raise RuntimeError("weight_mask must not require grad")

        effective_weight = ops.mul(self.weight, self.weight_mask)
        return ops.add(ops.matmul(x, effective_weight), self.bias)

    def parameters(self):
        # The mask is deliberately excluded: it must never reach the optimizer.
        return [self.weight, self.bias]

    def __repr__(self):
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, init={self.init!r}, "
            f"active_weights={self.active_weights}/{self.weight.data.size})"
        )