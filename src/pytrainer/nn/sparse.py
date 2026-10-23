"""Inference paths that actually skip pruned connections.

Training uses ``Linear``'s dense ``x @ (W * M)``: every pruned connection is
still multiplied, by zero, so a pruned model trained that way costs exactly
as much as the dense one. The two paths here do less work:

* ``SparseLayer``: stores only the kept connections and performs one
  multiply-add per kept connection per sample.
* ``compact``: deletes hidden units that pruning disconnected, leaving a
  smaller *dense* network with the same outputs.

Both take plain ``[(W * M, b), ...]`` arrays from ``masked_layers``; they
are inference-only (no autograd).
"""

import numpy as np


def masked_layers(model):
    """[(W * M, b), ...] per layer; pruned entries are exact zeros."""
    return [
        (layer.weight.data * layer.weight_mask.data, layer.bias.data.copy())
        for layer in (getattr(model, name) for name in model.LAYERS)
    ]


def forward(layers, x):
    """ReLU MLP forward over layers that are callables ``x -> x @ W + b``."""
    for layer in layers[:-1]:
        x = np.maximum(layer(x), 0.0)
    return layers[-1](x)


def dense(w, b):
    return lambda x: x @ w + b


def macs(layers):
    """Multiply-adds per sample of the dense ``(W, b)`` layers."""
    return sum(w.size for w, _ in layers)


class SparseLayer:
    """``x @ W + b`` touching only the nonzero entries of W (CSC order)."""

    def __init__(self, w, b):
        cols, rows = np.nonzero(w.T)  # kept connections, grouped by output unit
        self.rows = rows
        self.vals = w[rows, cols]
        self.b = b
        self.out_units, self.starts = np.unique(cols, return_index=True)
        self.nnz = rows.size  # multiply-adds per sample

    def __call__(self, x):
        y = np.empty((len(x), self.b.size))
        y[:] = self.b
        if self.nnz:
            products = x[:, self.rows] * self.vals  # one multiply per kept connection
            y[:, self.out_units] += np.add.reduceat(products, self.starts, axis=1)
        return y


def compact(layers):
    """Remove hidden units that pruning disconnected; outputs are unchanged.

    A unit with no outgoing connections cannot affect the output. A unit with
    no incoming connections outputs the constant relu(b), which is folded into
    the next layer's bias. Repeats until stable, since removing a unit can
    disconnect others.
    """
    layers = [(w.copy(), b.copy()) for w, b in layers]
    changed = True
    while changed:
        changed = False
        for k in range(len(layers) - 1):
            (w, b), (w_next, b_next) = layers[k], layers[k + 1]
            no_out = ~w_next.any(axis=1)
            constant = ~w.any(axis=0) & ~no_out
            keep = w.any(axis=0) & ~no_out
            if keep.all():
                continue
            b_next = b_next + np.maximum(b[constant], 0.0) @ w_next[constant]
            layers[k] = (w[:, keep], b[keep])
            layers[k + 1] = (w_next[keep], b_next)
            changed = True
    return layers
