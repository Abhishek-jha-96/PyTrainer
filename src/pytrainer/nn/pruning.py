"""Global Taylor-saliency pruning with an exact connection budget.

Only weight matrices are eligible. Biases are never pruned, so ``sparsity()``
is the fraction of weight connections removed, not of all trainable
parameters.

The layer's mask Tensor (``Linear.weight_mask``) is the single source of
truth for which connections exist. The pruner reads masks from those Tensors
and writes them back in place.
"""

import numpy as np

CRITERIA = ("taylor_abs", "taylor_sq", "magnitude")


def cubic_sparsity(epoch, start_epoch, end_epoch, target=0.90):
    """s(t) = s_f * [1 - (1 - u)^3], u = clip((t - t0) / (t1 - t0), 0, 1)."""
    if not 0.0 <= target < 1.0:
        raise ValueError("target must be in [0, 1).")
    if end_epoch <= start_epoch:
        raise ValueError("end_epoch must exceed start_epoch.")
    progress = np.clip((epoch - start_epoch) / (end_epoch - start_epoch), 0.0, 1.0)
    return float(target * (1.0 - (1.0 - progress) ** 3))


def removal_budget(target_sparsity, total):
    """Exact number of connections to remove, rounded half up."""
    return int(np.floor(target_sparsity * total + 0.5))


class GlobalTaylorPruner:
    """Rank all eligible weights together and prune to an exact count.

    Parameters
    ----------
    named_weights : mapping name -> weight Tensor (``.data``, ``.grad``).
        Insertion order fixes the deterministic tie-breaking order.
    mask_tensors : mapping name -> non-trainable mask Tensor of the same
        shape, holding only 0 and 1. Updated in place.
    ema_decay : EMA decay per observed minibatch.
    criterion : "taylor_abs" |w*g|, "taylor_sq" (w*g)^2, or "magnitude" |w|.
    min_keep_fraction : each layer keeps at least ceil(f * layer size)
        weights (0 disables the floor).
    output_name : optional name of the output weight, shape (in, out);
        every output unit then keeps at least ``min_per_output_unit`` inputs
        (or all it still has, if fewer are alive).
    """

    def __init__(
        self,
        named_weights,
        mask_tensors,
        ema_decay=0.90,
        criterion="taylor_abs",
        min_keep_fraction=0.05,
        output_name=None,
        min_per_output_unit=1,
    ):
        self.weights = dict(named_weights)
        self.mask_tensors = dict(mask_tensors)
        self.names = list(self.weights)

        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}.")
        if not 0.0 <= min_keep_fraction <= 1.0:
            raise ValueError("min_keep_fraction must be in [0, 1].")
        if not self.names:
            raise ValueError("No eligible weights.")
        if set(self.weights) != set(self.mask_tensors):
            raise ValueError("Weights and masks must have matching keys.")
        if min_per_output_unit < 0:
            raise ValueError("min_per_output_unit must be >= 0.")
        if output_name is not None and output_name not in self.weights:
            raise ValueError(f"Unknown output_name {output_name!r}.")
        for name in self.names:
            m = self.mask_tensors[name].data
            if self.weights[name].data.shape != m.shape:
                raise ValueError(f"Mask shape mismatch for {name}.")
            if not np.all((m == 0) | (m == 1)):
                raise ValueError(f"Mask for {name} must contain only 0 and 1.")

        self.ema_decay = ema_decay
        self.criterion = criterion
        self.min_keep_fraction = min_keep_fraction
        self.output_name = output_name
        self.min_per_output_unit = int(min_per_output_unit)

        self.scores = {n: np.zeros(self.weights[n].data.shape) for n in self.names}
        self._steps = dict.fromkeys(self.names, 0)
        self.safeguard_kept = 0

    # ------------------------------------------------------------------ masks
    def mask(self, name):
        """Boolean mask read from the layer's mask Tensor (a fresh array)."""
        return self.mask_tensors[name].data != 0

    @property
    def masks(self):
        return {name: self.mask(name) for name in self.names}

    # --------------------------------------------------------------- saliency
    def observe(self):
        """Update scores. Call after backward() and before optimizer.step()."""
        if self.criterion == "magnitude":
            return
        for name in self.names:
            param = self.weights[name]
            if param.grad is None:
                continue
            wg = param.data * param.grad  # abs/square per minibatch, before averaging
            current = np.abs(wg) if self.criterion == "taylor_abs" else wg * wg
            score = self.scores[name]
            score *= self.ema_decay
            score += (1.0 - self.ema_decay) * np.where(self.mask(name), current, 0.0)
            self._steps[name] += 1

    def saliency(self, name):
        """Bias-corrected EMA score (or |w| for the magnitude criterion)."""
        if self.criterion == "magnitude":
            return np.abs(self.weights[name].data)
        t = self._steps[name]
        if t == 0:
            return np.zeros_like(self.scores[name])
        return self.scores[name] / (1.0 - self.ema_decay ** t)

    # ---------------------------------------------------------------- pruning
    def _protected(self, name, mask, score):
        """Boolean array (mask's shape) of alive weights kept by safeguards."""
        protected = np.zeros(mask.shape, dtype=bool)

        # Per-layer floor: the top-scoring alive weights.
        alive = np.flatnonzero(mask)
        floor = min(int(np.ceil(self.min_keep_fraction * mask.size)), alive.size)
        order = np.argsort(-score.ravel()[alive], kind="stable")
        protected.flat[alive[order[:floor]]] = True

        # Output units: each keeps its top-scoring alive inputs.
        if name == self.output_name:
            for j in range(mask.shape[1]):
                rows = np.flatnonzero(mask[:, j])
                top = rows[np.argsort(-score[rows, j], kind="stable")]
                protected[top[: self.min_per_output_unit], j] = True
        return protected

    def prune_to(self, target_sparsity, optimizer=None):
        """Prune the lowest-scoring alive weights to an exact global count.

        Monotonic: pruned weights never regrow. Returns the new sparsity.
        """
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1).")

        removed, total = self._counts()
        number_to_remove = removal_budget(target_sparsity, total) - removed
        if number_to_remove < 0:
            raise ValueError("Target sparsity cannot decrease for a monotonic pruner.")

        self.safeguard_kept = 0
        if number_to_remove:
            if self.criterion != "magnitude" and not any(self._steps.values()):
                raise ValueError("No saliency observed; call observe() before pruning.")
            for name, new_mask in self._select(number_to_remove).items():
                target = self.mask_tensors[name].data
                target[...] = new_mask  # in place: the layer's mask Tensor keeps its identity

        self.enforce(optimizer)
        return self.sparsity()

    def _select(self, number_to_remove):
        """New masks with the lowest-scoring unprotected alive weights removed."""
        masks = self.masks
        owners, indices, scores, protected = [], [], [], []
        for k, name in enumerate(self.names):
            mask, score = masks[name], self.saliency(name)
            alive = np.flatnonzero(mask)
            owners.append(np.full(alive.size, k))
            indices.append(alive)
            scores.append(score.ravel()[alive])
            protected.append(self._protected(name, mask, score).ravel()[alive])

        owners = np.concatenate(owners)
        indices = np.concatenate(indices)
        scores = np.concatenate(scores)
        protected = np.concatenate(protected)

        candidates = np.flatnonzero(~protected)
        if number_to_remove > candidates.size:
            raise ValueError(
                f"Cannot remove {number_to_remove} weights: only "
                f"{candidates.size} are alive and not held by safeguards."
            )

        # Stable sort: ties break by parameter order, then flat index.
        order = np.argsort(scores[candidates], kind="stable")
        selected = candidates[order[:number_to_remove]]

        # Diagnostic: survivors that exist only because of the safeguards.
        unconstrained = np.argsort(scores, kind="stable")[:number_to_remove]
        self.safeguard_kept = int(np.count_nonzero(protected[unconstrained]))

        for k, name in enumerate(self.names):
            masks[name].flat[indices[selected[owners[selected] == k]]] = False
        return masks

    # ------------------------------------------------------------ enforcement
    def enforce(self, optimizer=None):
        """Zero pruned weights and gradients; hand each mask to a MaskedAdam."""
        for name in self.names:
            mask, param = self.mask(name), self.weights[name]
            param.data[~mask] = 0.0
            if param.grad is not None:
                param.grad[~mask] = 0.0
            if optimizer is not None:
                optimizer.set_mask(param, mask)

    def check_integrity(self, optimizer=None):
        """Raise unless pruned weights, gradients and (with a MaskedAdam)
        Adam moments are exactly 0, and the optimizer's mask matches."""
        for name in self.names:
            mask, param = self.mask(name), self.weights[name]
            if np.any(param.data[~mask] != 0):
                raise RuntimeError(f"Pruned raw weights of {name} are nonzero.")
            if param.grad is not None and np.any(param.grad[~mask] != 0):
                raise RuntimeError(f"Pruned gradients of {name} are nonzero.")
            if optimizer is None:
                continue
            first, second = optimizer.moments(param)
            if np.any(first[~mask] != 0) or np.any(second[~mask] != 0):
                raise RuntimeError(f"Adam moments of {name} are nonzero at pruned entries.")
            stored = optimizer.mask_of(param)
            if stored is not None and not np.array_equal(stored, mask):
                raise RuntimeError(f"Optimizer mask for {name} differs from the layer mask.")

    # -------------------------------------------------------------- reporting
    def _counts(self):
        """(removed, total) eligible connections."""
        masks = self.masks.values()
        return sum(int(np.count_nonzero(~m)) for m in masks), sum(m.size for m in masks)

    def sparsity(self):
        """Fraction of eligible weight connections removed (biases excluded)."""
        removed, total = self._counts()
        return removed / total

    def layer_report(self):
        """{name: {"kept", "total", "sparsity"}} per eligible weight."""
        report = {}
        for name, m in self.masks.items():
            kept, total = int(np.count_nonzero(m)), int(m.size)
            report[name] = {"kept": kept, "total": total, "sparsity": 1.0 - kept / total}
        return report
