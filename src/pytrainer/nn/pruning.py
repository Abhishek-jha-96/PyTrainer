"""Global Taylor-saliency pruning with an exact connection budget.

Only weight matrices are eligible. Biases are never pruned, so ``sparsity()``
is the fraction of weight connections removed, not of all trainable
parameters.

The layer's mask Tensor (``Linear.weight_mask``) is the single source of
truth for which connections exist. The pruner reads masks from those Tensors
and writes them back in place; it keeps no second copy that could drift.
"""

import numpy as np

CRITERIA = ("taylor_abs", "taylor_sq", "magnitude")


def cubic_sparsity(epoch, start_epoch, end_epoch, target=0.90):
    """s(t) = s_f * [1 - (1 - u)^3], u = clip((t - t0) / (t1 - t0), 0, 1)."""
    if not 0.0 <= target < 1.0:
        raise ValueError("target must be in [0, 1).")
    if end_epoch <= start_epoch:
        raise ValueError("end_epoch must exceed start_epoch.")
    if epoch <= start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return float(target)
    progress = (epoch - start_epoch) / (end_epoch - start_epoch)
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
        named_weights = dict(named_weights)
        mask_tensors = dict(mask_tensors)

        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}.")
        if not 0.0 <= min_keep_fraction <= 1.0:
            raise ValueError("min_keep_fraction must be in [0, 1].")
        if not named_weights:
            raise ValueError("No eligible weights.")
        if set(named_weights) != set(mask_tensors):
            raise ValueError("Weights and masks must have matching keys.")
        if min_per_output_unit < 0:
            raise ValueError("min_per_output_unit must be >= 0.")
        if output_name is not None and output_name not in named_weights:
            raise ValueError(f"Unknown output_name {output_name!r}.")

        self.names = list(named_weights)
        self.weights = named_weights
        self.mask_tensors = mask_tensors
        self.ema_decay = ema_decay
        self.criterion = criterion
        self.min_keep_fraction = min_keep_fraction
        self.output_name = output_name
        self.min_per_output_unit = int(min_per_output_unit)

        for name in self.names:
            w = self.weights[name].data
            m = self.mask_tensors[name].data
            if w.shape != m.shape:
                raise ValueError(f"Mask shape mismatch for {name}.")
            if not np.all((m == 0) | (m == 1)):
                raise ValueError(f"Mask for {name} must contain only 0 and 1.")
        if output_name is not None and self.weights[output_name].data.ndim != 2:
            raise ValueError("output_name must refer to a 2-D weight.")

        self.scores = {
            name: np.zeros(self.weights[name].data.shape, dtype=np.float64)
            for name in self.names
        }
        self._steps = {name: 0 for name in self.names}
        self._expected = {name: self.mask(name).copy() for name in self.names}
        self.safeguard_kept = 0

    # ------------------------------------------------------------------ masks
    def mask(self, name):
        """Boolean mask read from the layer's mask Tensor (a fresh array)."""
        return np.asarray(self.mask_tensors[name].data) != 0

    @property
    def masks(self):
        return {name: self.mask(name) for name in self.names}

    def _write_mask(self, name, new_mask):
        target = self.mask_tensors[name].data
        target[...] = new_mask.astype(target.dtype)  # in place: identity kept
        self._expected[name] = new_mask.copy()

    # --------------------------------------------------------------- saliency
    def observe(self):
        """Update scores. Call after backward() and before optimizer.step()."""
        if self.criterion == "magnitude":
            return
        for name in self.names:
            param = self.weights[name]
            if param.grad is None:
                continue
            w = np.asarray(param.data, dtype=np.float64)
            g = np.asarray(param.grad, dtype=np.float64)
            if g.shape != w.shape:
                raise ValueError(f"Gradient shape mismatch for {name}.")
            if not np.all(np.isfinite(g)):
                raise FloatingPointError(f"Non-finite gradient in {name}.")

            wg = w * g  # per-minibatch product; abs/square before averaging
            current = np.abs(wg) if self.criterion == "taylor_abs" else wg * wg
            current = np.where(self.mask(name), current, 0.0)

            score = self.scores[name]
            score *= self.ema_decay
            score += (1.0 - self.ema_decay) * current
            self._steps[name] += 1

    def saliency(self, name):
        """Bias-corrected EMA score (or |w| for the magnitude criterion)."""
        if self.criterion == "magnitude":
            return np.abs(np.asarray(self.weights[name].data, dtype=np.float64))
        t = self._steps[name]
        if t == 0:
            return np.zeros_like(self.scores[name])
        return self.scores[name] / (1.0 - self.ema_decay ** t)

    # ---------------------------------------------------------------- pruning
    def _protected(self, name, mask, score):
        """Flat (C-order) boolean array of alive weights kept by safeguards."""
        flat_mask = mask.reshape(-1)
        flat_score = score.reshape(-1)
        protected = np.zeros(flat_mask.size, dtype=bool)
        alive = np.flatnonzero(flat_mask)

        floor = int(np.ceil(self.min_keep_fraction * flat_mask.size))
        floor = min(floor, alive.size)
        if floor:
            order = np.argsort(-flat_score[alive], kind="stable")
            protected[alive[order[:floor]]] = True

        if name == self.output_name and self.min_per_output_unit:
            for j in range(mask.shape[1]):
                rows = np.flatnonzero(mask[:, j])
                top = rows[np.argsort(-score[rows, j], kind="stable")]
                for i in top[: self.min_per_output_unit]:
                    protected[np.ravel_multi_index((i, j), mask.shape)] = True
        return protected

    def prune_to(self, target_sparsity, optimizer=None):
        """Prune the lowest-scoring alive weights to an exact global count.

        Monotonic: pruned weights never regrow. Returns the new sparsity.
        """
        if not 0.0 <= target_sparsity < 1.0:
            raise ValueError("target_sparsity must be in [0, 1).")

        masks = self.masks
        total = sum(m.size for m in masks.values())
        target_removed = removal_budget(target_sparsity, total)
        already_removed = sum(int(np.count_nonzero(~m)) for m in masks.values())
        if target_removed < already_removed:
            raise ValueError("Target sparsity cannot decrease for a monotonic pruner.")

        number_to_remove = target_removed - already_removed
        self.safeguard_kept = 0

        if number_to_remove:
            if self.criterion != "magnitude" and not any(self._steps.values()):
                raise ValueError("No saliency observed; call observe() before pruning.")
            new_masks = self._select(masks, number_to_remove)
            for name in self.names:
                self._write_mask(name, new_masks[name])

        self.enforce(optimizer)
        return self.sparsity()

    def _select(self, masks, number_to_remove):
        owners, indices, scores, protected = [], [], [], []
        for k, name in enumerate(self.names):
            mask = masks[name]
            score = self.saliency(name)
            alive = np.flatnonzero(mask.reshape(-1))
            owners.append(np.full(alive.size, k))
            indices.append(alive)
            scores.append(score.reshape(-1)[alive])
            protected.append(self._protected(name, mask, score)[alive])

        owners = np.concatenate(owners)
        indices = np.concatenate(indices)
        scores = np.concatenate(scores)
        protected = np.concatenate(protected)

        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("Non-finite saliency scores.")

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

        new_masks = {}
        for k, name in enumerate(self.names):
            flat = masks[name].reshape(-1).copy()  # explicit copy, no view reliance
            flat[indices[selected[owners[selected] == k]]] = False
            new_masks[name] = flat.reshape(masks[name].shape)
        return new_masks

    # ------------------------------------------------------------ enforcement
    def enforce(self, optimizer=None):
        """Zero pruned weights and gradients; push masks to MaskedAdam."""
        for name in self.names:
            mask = self.mask(name)
            param = self.weights[name]
            param.data[~mask] = 0.0
            if param.grad is not None:
                param.grad[~mask] = 0.0
            if optimizer is not None:
                if not hasattr(optimizer, "set_mask"):
                    raise TypeError("Optimizer must implement set_mask(param, mask).")
                optimizer.set_mask(param, mask)

    def check_integrity(self):
        """Raise if a mask Tensor changed outside prune_to, or a pruned raw
        weight is nonzero."""
        for name in self.names:
            m = np.asarray(self.mask_tensors[name].data)
            if not np.all((m == 0) | (m == 1)):
                raise ValueError(f"Mask for {name} contains values other than 0 and 1.")
            mask = m != 0
            if not np.array_equal(mask, self._expected[name]):
                raise RuntimeError(f"Mask for {name} was modified outside the pruner.")
            if np.any(np.asarray(self.weights[name].data)[~mask] != 0):
                raise RuntimeError(f"Pruned raw weights of {name} are nonzero.")

    # -------------------------------------------------------------- reporting
    def sparsity(self):
        """Fraction of eligible weight connections removed (biases excluded)."""
        masks = self.masks
        total = sum(m.size for m in masks.values())
        removed = sum(int(np.count_nonzero(~m)) for m in masks.values())
        return removed / total

    def layer_report(self):
        """{name: {"kept", "total", "sparsity"}} per eligible weight."""
        report = {}
        for name, m in self.masks.items():
            kept, total = int(np.count_nonzero(m)), int(m.size)
            report[name] = {"kept": kept, "total": total,
                            "sparsity": 1.0 - kept / total}
        return report