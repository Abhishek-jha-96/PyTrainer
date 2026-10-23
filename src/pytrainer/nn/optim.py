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

    Epsilon convention: PyTorch. ``eps = 1e-8`` is added *after* the
    square root of the bias-corrected second moment, exactly as in the
    formula above. (Keras defaults to ``epsilon = 1e-7`` and applies it
    to the uncorrected ``sqrt(v_t)``; that convention is not used here.)

    Defaults match PyTorch: lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8.

    Gradient handling:
        * ``zero_grad()`` sets every ``param.grad`` to ``None``. The
          engine lazily recreates a zero array on the next backward pass.
        * A parameter whose ``grad`` is ``None`` at ``step()`` time is
          skipped: its value and moments are left unchanged (PyTorch's
          behaviour). The shared step counter still advances.
        * A zero gradient is a real gradient: moments are updated. With
          zero moments the update is exactly 0, so the parameter does not
          move; with nonzero moments from earlier steps, momentum still
          moves it, as in standard Adam.
    """

    def __init__(
        self,
        parameters,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
    ):
        params = list(parameters)
        if not params:
            raise ValueError("Adam received an empty parameter list")

        seen = set()
        for i, p in enumerate(params):
            if not isinstance(p, Tensor):
                raise TypeError(f"parameter {i} is {type(p).__name__}, not Tensor")
            if not p.requires_grad:
                raise ValueError(f"parameter {i} does not require grad")
            if id(p) in seen:
                raise ValueError(f"parameter {i} appears more than once")
            seen.add(id(p))

        if not lr > 0.0:
            raise ValueError(f"lr must be > 0, got {lr}")
        if not 0.0 <= beta1 < 1.0:
            raise ValueError(f"beta1 must be in [0, 1), got {beta1}")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError(f"beta2 must be in [0, 1), got {beta2}")
        if not eps > 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        # References to the model's own Tensors (not copies).
        self.parameters = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # Persistent per-parameter state; lives for the optimizer's lifetime.
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

        # One counter per optimizer step, shared by all parameters.
        self.t = 0

    def zero_grad(self):
        """Clear gradients before the next minibatch's backward pass."""
        for p in self.parameters:
            p.grad = None

    def step(self):
        """Apply one Adam update to every parameter that has a gradient."""
        self.t += 1
        t = self.t
        b1, b2 = self.beta1, self.beta2

        # Scalar bias corrections, shared across parameters for this step.
        bias_c1 = 1.0 - b1**t
        bias_c2 = 1.0 - b2**t

        for i, (p, m, v) in enumerate(zip(self.parameters, self.m, self.v)):
            g = p.grad
            if g is None:
                continue

            g = np.asarray(g, dtype=np.float64)
            if g.shape != p.data.shape:
                raise ValueError(
                    f"parameter {i}: grad shape {g.shape} != "
                    f"data shape {p.data.shape}"
                )
            if not np.all(np.isfinite(g)):
                raise FloatingPointError(f"parameter {i}: non-finite gradient")

            # Moment updates, in place so the state arrays persist.
            m *= b1
            m += (1.0 - b1) * g
            v *= b2
            v += (1.0 - b2) * (g * g)

            m_hat = m / bias_c1
            v_hat = v / bias_c2
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # In place on the existing array: the Tensor object, and the
            # ndarray it holds, keep their identity.
            np.subtract(p.data, update, out=p.data)

    def __repr__(self):
        return (
            f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, "
            f"eps={self.eps}, params={len(self.parameters)}, t={self.t})"
        )


class MaskedAdam(Adam):
    """Adam that keeps pruned weights exactly at zero.
 
    For a masked-out entry the optimizer:
      * resets its first and second moments when the mask is set, so no
        stored momentum can move it later;
      * zeroes its gradient before every update;
      * re-applies the mask to the weight after every update.
 
    Unmasked parameters (e.g. biases) behave exactly like plain Adam.
    """
 
    def __init__(self, parameters, **kwargs):
        super().__init__(parameters, **kwargs)
        self._index = {id(p): i for i, p in enumerate(self.parameters)}
        self.masks = [None] * len(self.parameters)
 
    def set_mask(self, param, mask):
        i = self._index.get(id(param))
        if i is None:
            raise ValueError("parameter is not managed by this optimizer")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != param.data.shape:
            raise ValueError(f"mask shape {mask.shape} != parameter shape {param.data.shape}")
        self.masks[i] = mask
        self.m[i][~mask] = 0.0
        self.v[i][~mask] = 0.0
        param.data[~mask] = 0.0
 
    def step(self):
        for p, mask in zip(self.parameters, self.masks):
            if mask is not None and p.grad is not None:
                p.grad = np.where(mask, p.grad, 0.0)
        super().step()
        for p, mask in zip(self.parameters, self.masks):
            if mask is not None:
                p.data[~mask] = 0.0
 
