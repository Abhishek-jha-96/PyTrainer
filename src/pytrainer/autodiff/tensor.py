import numpy as np


class Tensor:
    """A minimal Tensor wrapping a NumPy array with autograd metadata."""

    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        if isinstance(data, Tensor):
            data = data.data
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None  # lazily initialized to zeros on first use
        self._parents = tuple(_parents)
        self._backward = _backward

    # ------------------------------------------------------------------
    # Shape / dimension / representation helpers
    # ------------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def __len__(self):
        return len(self.data)

    def item(self):
        return self.data.item()

    def __repr__(self):
        rg = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({np.array2string(self.data, precision=4)}{rg})"

    def _ensure_grad(self):
        """Initialize grad storage (zero-valued) if not yet present."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        return self.grad

    def _graph_metadata(self):
        """Inspectable graph metadata for this node."""
        return {
            "id": id(self),
            "shape": self.shape,
            "requires_grad": self.requires_grad,
            "grad": self.grad,
            "parents": [id(p) for p in self._parents],
            "has_backward": self._backward is not None,
        }

    # ------------------------------------------------------------------
    # Backward pass: reverse topological traversal of the graph
    # ------------------------------------------------------------------
    def backward(self, seed=None):
        if not self.requires_grad:
            raise RuntimeError("called backward on a tensor that does not require grad")
        
        # initialize the output gradient
        if seed is None:
            if self.data.ndim != 0:
                raise ValueError(
                    "seed must be provided for non-scalar outputs"
                )
            seed = np.ones_like(self.data)
        else:
            seed = np.asarray(seed, dtype=np.float64)
            if seed.shape != self.data.shape:
                raise ValueError(
                    f"Expected seed shape {self.data.shape}, "
                    f"got {seed.shape}"
                )
            
        self._ensure_grad()
        self.grad += seed

        topo, state, stack = [], {}, [(self, False)]
        while stack:
            node, expanded = stack.pop()
            node_id = id(node)

            if expanded:
                if state.get(node_id) == 2:
                    continue

                state[node_id] = 2
                topo.append(node)
                continue

            if state.get(node_id) == 2:
                continue

            if state.get(node_id) == 1:
                continue

            state[node_id] = 1
            stack.append((node, True))

            for parent in reversed(node._parents):
                if state.get(id(parent)) != 2:
                    stack.append((parent, False))

        for node in reversed(topo):
            if node._backward is not None:
                node._backward()


# Attach differentiable operations as dunder methods for syntactic sugar
from pytrainer.autodiff import ops
Tensor.__add__ = lambda self, other: ops.add(self, other)
Tensor.__radd__ = lambda self, other: ops.add(other, self)
Tensor.__mul__ = lambda self, other: ops.mul(self, other)
Tensor.__rmul__ = lambda self, other: ops.mul(other, self)
Tensor.__sub__ = lambda self, other: ops.sub(self, other)
Tensor.__rsub__ = lambda self, other: ops.sub(other, self)
Tensor.__truediv__ = lambda self, other: ops.div(self, other)
Tensor.__rtruediv__ = lambda self, other: ops.div(other, self)
Tensor.__pow__ = lambda self, other: ops.pow(self, other)
Tensor.__getitem__ = lambda self, indices: ops.getitem(self, indices)
Tensor.__matmul__ = lambda self, other: ops.matmul(self, other)
Tensor.__rmatmul__ = lambda self, other: ops.matmul(other, self)
