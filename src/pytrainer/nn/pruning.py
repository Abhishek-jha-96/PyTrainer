"""Global saliency pruning for Linear weights.

Components:
    SaliencyTracker  per-weight EMA of a pruning score, updated every minibatch
    cubic_sparsity   gradual sparsity schedule (Zhu & Gupta, 2017)
    select_masks     global top-k selection with per-layer safeguards
    Pruner           ties the three together and owns the binary masks

Scores (``criterion``):
    "taylor_abs"  |w * g_b|      first-order Taylor estimate of the loss change
    "taylor_sq"   (w * g_b)**2   averaged: w^2 * diag(empirical Fisher), an
                                 OBD-style second-order saliency
    "magnitude"   |w|            baseline

``g_b`` is the gradient of the *minibatch* mean loss. The absolute value
(or square) is taken per minibatch and only then averaged by the EMA:
averaging gradients first would collapse the scores toward the
near-zero full-dataset gradient of a trained network.

Only weights are pruned; biases stay dense. Masks are monotone: a pruned
connection never regrows.
"""

import numpy as np

CRITERIA = ("taylor_abs", "taylor_sq", "magnitude")


# ----------------------------------------------------------------------
# Scores
# ----------------------------------------------------------------------
def instantaneous_saliency(weight, grad, criterion):
    if criterion == "taylor_abs":
        return np.abs(weight * grad)
    if criterion == "taylor_sq":
        return (weight * grad) ** 2
    if criterion == "magnitude":
        return np.abs(weight)
    raise ValueError(f"unknown criterion {criterion!r}; choose from {CRITERIA}")


class SaliencyTracker:
    """Bias-corrected exponential moving average of per-weight saliency.

    ``update()`` must be called after ``loss.backward()`` and before
    ``optimizer.step()``, so each weight is paired with the gradient
    computed at that weight.
    """

    def __init__(self, named_weights, criterion="taylor_abs", decay=0.9):
        if criterion not in CRITERIA:
            raise ValueError(f"unknown criterion {criterion!r}; choose from {CRITERIA}")
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self.named_weights = list(named_weights)
        self.criterion = criterion
        self.decay = decay
        self.ema = {name: np.zeros_like(w.data) for name, w in self.named_weights}
        self.updates = 0

    def update(self):
        self.updates += 1
        b = self.decay
        for name, w in self.named_weights:
            if w.grad is None and self.criterion != "magnitude":
                raise RuntimeError(f"{name} has no gradient; call after backward()")
            g = w.grad if w.grad is not None else np.zeros_like(w.data)
            s = instantaneous_saliency(w.data, g, self.criterion)
            ema = self.ema[name]
            ema *= b
            ema += (1.0 - b) * s

    def scores(self):
        """Bias-corrected scores (no start-up shrinkage toward zero)."""
        if self.updates == 0:
            raise RuntimeError("no saliency measurements yet")
        correction = 1.0 - self.decay**self.updates
        return {name: ema / correction for name, ema in self.ema.items()}


# ----------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------
def cubic_sparsity(event, num_events, initial=0.0, final=0.9):
    """Sparsity target at pruning event ``event`` (0 .. num_events).

        s_k = s_f + (s_i - s_f) * (1 - k / n)**3

    Front-loaded: prunes fast while many redundant weights remain and
    slows near the target. Clamped to ``final`` after the last event.
    """
    if num_events <= 0:
        raise ValueError("num_events must be positive")
    if not 0.0 <= initial <= final < 1.0:
        raise ValueError(f"need 0 <= initial <= final < 1, got {initial}, {final}")
    k = min(max(event, 0), num_events)
    return final + (initial - final) * (1.0 - k / num_events) ** 3


# ----------------------------------------------------------------------
# Mask selection
# ----------------------------------------------------------------------
def select_masks(scores, current_masks, sparsity, min_keep_fraction=0.05,
                 output_layer=None, min_per_output_unit=1):
    """Choose which weights survive, ranking all layers together.

    Steps:
      1. Only currently alive weights are candidates (monotone masks).
      2. Reserve per-layer floors: the top ``ceil(min_keep_fraction * size)``
         weights of every layer, and the top ``min_per_output_unit``
         incoming weights of every unit in ``output_layer`` (so every class
         logit keeps a connection).
      3. Fill the remaining budget with the globally highest scores.

    Returns new masks (bool arrays); total kept = round((1 - sparsity) * total),
    unless the floors alone require more.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")

    names = list(scores)
    total = sum(scores[n].size for n in names)
    budget = int(round((1.0 - sparsity) * total))

    keep = {n: np.zeros(scores[n].shape, dtype=bool) for n in names}
    ranked = {}
    for n in names:
        s = np.where(current_masks[n], scores[n], -np.inf)
        ranked[n] = s

    def take_top(n, flat_candidates, k):
        s = ranked[n].ravel()
        order = flat_candidates[np.argsort(-s[flat_candidates], kind="stable")]
        order = order[np.isfinite(s[order])][:k]
        keep[n].ravel()[order] = True

    # 2a. per-layer floors
    for n in names:
        floor = int(np.ceil(min_keep_fraction * scores[n].size))
        take_top(n, np.arange(scores[n].size), floor)

    # 2b. per-output-unit guarantee (columns of a (fan_in, fan_out) weight)
    if output_layer is not None and min_per_output_unit > 0:
        s = ranked[output_layer]
        for j in range(s.shape[1]):
            col = np.ravel_multi_index((np.arange(s.shape[0]), np.full(s.shape[0], j)), s.shape)
            take_top(output_layer, col, min_per_output_unit)

    # 3. global fill
    reserved = sum(int(k.sum()) for k in keep.values())
    remaining = budget - reserved
    if remaining > 0:
        flat = np.concatenate([np.where(keep[n], -np.inf, ranked[n]).ravel() for n in names])
        owner = np.concatenate([np.full(scores[n].size, i) for i, n in enumerate(names)])
        local = np.concatenate([np.arange(scores[n].size) for n in names])
        order = np.argsort(-flat, kind="stable")
        order = order[np.isfinite(flat[order])][:remaining]
        for i, n in enumerate(names):
            keep[n].ravel()[local[order[owner[order] == i]]] = True

    return {n: keep[n] & current_masks[n] for n in names}


# ----------------------------------------------------------------------
# Pruner
# ----------------------------------------------------------------------
class Pruner:
    """Owns the weight masks and applies them to the model and optimizer."""

    def __init__(self, named_weights, optimizer, criterion="taylor_abs", decay=0.9,
                 min_keep_fraction=0.05, output_layer=None, min_per_output_unit=1):
        self.named_weights = list(named_weights)
        self.weights = dict(self.named_weights)
        self.optimizer = optimizer
        self.tracker = SaliencyTracker(self.named_weights, criterion, decay)
        self.min_keep_fraction = min_keep_fraction
        self.output_layer = output_layer
        self.min_per_output_unit = min_per_output_unit
        self.masks = {n: np.ones(w.shape, dtype=bool) for n, w in self.named_weights}
        for n, w in self.named_weights:
            optimizer.set_mask(w, self.masks[n])

    def update_scores(self):
        self.tracker.update()

    def prune_to(self, sparsity):
        new = select_masks(
            self.tracker.scores(), self.masks, sparsity,
            self.min_keep_fraction, self.output_layer, self.min_per_output_unit,
        )
        self.masks = new
        for n, w in self.named_weights:
            self.optimizer.set_mask(w, new[n])  # zeroes weights and resets moments
        return self.sparsity()

    def sparsity(self):
        kept = sum(int(m.sum()) for m in self.masks.values())
        total = sum(m.size for m in self.masks.values())
        return 1.0 - kept / total

    def layer_report(self):
        """Per-layer kept counts and unit connectivity diagnostics."""
        names = [n for n, _ in self.named_weights]
        out = {}
        for i, n in enumerate(names):
            m = self.masks[n]
            info = {"kept": int(m.sum()), "total": m.size,
                    "sparsity": 1.0 - m.sum() / m.size}
            # Units produced by this layer: column j has no incoming weights.
            info["units_without_inputs"] = int(np.sum(~m.any(axis=0)))
            if i + 1 < len(names):
                nxt = self.masks[names[i + 1]]
                # Units whose outgoing row in the next layer is fully pruned.
                info["units_without_outputs"] = int(np.sum(~nxt.any(axis=1)))
            out[n] = info
        return out