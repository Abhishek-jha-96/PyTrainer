import numpy as np


class Tensor:
    """A minimal Tensor wrapping a NumPy array with autograd metadata.

    Design: each operation returns a NEW Tensor that records its parents
    (the input tensors) and a local backward function, so gradients are
    accumulated by traversing the graph in reverse topological order
    rather than by differentiating the whole expression at once.
    """

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

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def _ensure_grad(self):
        """Initialize grad storage (zero-valued) if not yet present."""
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        return self.grad

    def __repr__(self):
        rg = ", requires_grad=True" if self.requires_grad else ""
        has_graph = ", graph" if (self._parents or self._backward) else ""
        return f"Tensor({np.array2string(self.data, precision=4)}{rg}{has_graph})"

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
    # Operations: each builds a new Tensor and captures a local backward rule
    # ------------------------------------------------------------------
    @staticmethod
    def _from_op(data, parents, backward):
        """Result tensor tracks grads iff any parent does."""
        requires_grad = any(p.requires_grad for p in parents)
        return Tensor(data, requires_grad=requires_grad,
                      _parents=parents, _backward=backward)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data + other.data

        def backward():
            g = out.grad
            if self.requires_grad:
                self._ensure_grad(); self.grad += Tensor._unbroadcast(g, self.shape)
            if other.requires_grad:
                other._ensure_grad(); other.grad += Tensor._unbroadcast(g, other.shape)

        out = Tensor._from_op(out_data, (self, other), backward)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out_data = self.data * other.data

        def backward():
            g = out.grad
            if self.requires_grad:
                self._ensure_grad(); self.grad += Tensor._unbroadcast(g * other.data, self.shape)
            if other.requires_grad:
                other._ensure_grad(); other.grad += Tensor._unbroadcast(g * self.data, other.shape)

        out = Tensor._from_op(out_data, (self, other), backward)
        return out


    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    

    # ------------------------------------------------------------------
    # Backward pass: reverse topological traversal of the graph
    # ------------------------------------------------------------------
    def backward(self, seed=None):
        if not self.requires_grad:
            raise RuntimeError("called backward on a tensor that does not require grad")
        if seed is None:
            seed = np.ones_like(self.data)
        # self._ensure_grad()[:] += seed
        self._ensure_grad()
        self.grad += seed

        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))

        for node in reversed(topo):
            if node._backward is not None:
                node._backward()
