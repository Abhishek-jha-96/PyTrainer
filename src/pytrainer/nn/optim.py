import numpy as np

from pytrainer.autodiff.tensor import Tensor


class Adam:
    """Adam optimizer (Kingma & Ba, 2015), no weight decay.

    For each parameter, with shared step counter ``t``::

        m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t**2
        m_hat_t = m_t / (1 - beta1**t)
        v_hat_t = v_t / (1 - beta2**t)
        theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)

    Defaults and the epsilon placement (after the square root of the
    bias-corrected second moment) follow PyTorch.

    ``zero_grad()`` sets every ``param.grad`` to ``None``; a parameter whose
    grad is ``None`` at ``step()`` is skipped (value and moments unchanged),
    while the shared step counter still advances.
    """

    def __init__(self, parameters, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        params = list(parameters)
        if not params:
            raise ValueError("Adam received an empty parameter list")
        for i, p in enumerate(params):
            if not isinstance(p, Tensor):
                raise TypeError(f"parameter {i} is {type(p).__name__}, not Tensor")
            if not p.requires_grad:
                raise ValueError(f"parameter {i} does not require grad")
        if len({id(p) for p in params}) != len(params):
            raise ValueError("a parameter appears more than once")
        if not lr > 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
            raise ValueError(f"betas must be in [0, 1), got {beta1}, {beta2}")
        if not eps > 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        # References to the model's own Tensors (not copies).
        self.parameters = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self._index = {id(p): i for i, p in enumerate(params)}
        self.t = 0

    def _slot(self, param):
        i = self._index.get(id(param))
        if i is None:
            raise ValueError("parameter is not managed by this optimizer")
        return i

    def moments(self, param):
        """The live (m, v) state arrays for ``param`` (not copies)."""
        i = self._slot(param)
        return self.m[i], self.v[i]

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    def step(self):
        """Apply one Adam update to every parameter that has a gradient."""
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bias_c1 = 1.0 - b1**self.t
        bias_c2 = 1.0 - b2**self.t

        for i, (p, m, v) in enumerate(zip(self.parameters, self.m, self.v)):
            if p.grad is None:
                continue
            g = np.asarray(p.grad, dtype=np.float64)
            if g.shape != p.shape:
                raise ValueError(f"parameter {i}: grad shape {g.shape} != data shape {p.shape}")
            if not np.all(np.isfinite(g)):
                raise FloatingPointError(f"parameter {i}: non-finite gradient")

            # In place, so the state arrays and the parameter keep their identity.
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)
            p.data -= self.lr * (m / bias_c1) / (np.sqrt(v / bias_c2) + self.eps)

    def __repr__(self):
        return (
            f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, "
            f"eps={self.eps}, params={len(self.parameters)}, t={self.t})"
        )


class MaskedAdam(Adam):
    """Adam that keeps pruned weights exactly at zero on every step.

    ``set_mask`` zeroes the pruned weights, gradients and moments once.
    ``step`` then zeroes the gradient at pruned entries before the update,
    so m, v and the update stay exactly 0 there. (Zeroing only the weight
    is not enough: a nonzero m_{t-1} moves it even when g_t = 0.)

    The mask is a boolean array owned by the optimizer, never a parameter.
    Unmasked parameters (e.g. biases) behave exactly like plain Adam.
    """

    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self.masks = [None] * len(self.parameters)

    def set_mask(self, param, mask):
        """Install a keep-mask (True = connection exists) for ``param``."""
        i = self._slot(param)
        mask = np.array(mask, dtype=bool)  # own copy: callers can't alias it
        if mask.shape != param.shape:
            raise ValueError(f"mask shape {mask.shape} != parameter shape {param.shape}")
        self.masks[i] = mask
        param.data[~mask] = 0.0
        self.m[i][~mask] = 0.0
        self.v[i][~mask] = 0.0
        if param.grad is not None:
            param.grad = np.where(mask, param.grad, 0.0)

    def mask_of(self, param):
        """The stored keep-mask for ``param``, or None if it is unmasked."""
        return self.masks[self._slot(param)]

    def step(self):
        for p, mask in zip(self.parameters, self.masks):
            if mask is not None and p.grad is not None:
                p.grad = np.where(mask, p.grad, 0.0)
        super().step()
