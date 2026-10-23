from pytrainer.autodiff import ops
from pytrainer.nn.layers import Linear, he_normal, xavier_uniform


class MLP:
    """2 -> 32 -> 16 -> 2 multilayer perceptron with ReLU activations.

    The final layer returns raw logits (no softmax). Normalization is
    handled inside the stable softmax cross-entropy loss.
    """

    LAYERS = ("fc1", "fc2", "fc3")

    def __init__(self, rng):
        self.fc1 = Linear(2, 32, rng, init=he_normal)
        self.fc2 = Linear(32, 16, rng, init=he_normal)
        self.fc3 = Linear(16, 2, rng, init=xavier_uniform)

    def __call__(self, x):
        x = ops.relu(self.fc1(x))
        x = ops.relu(self.fc2(x))
        return self.fc3(x)  # raw logits

    def named_parameters(self):
        """[("fc1.weight", W1), ("fc1.bias", b1), ...] in layer order."""
        named = []
        for name in self.LAYERS:
            layer = getattr(self, name)
            named += [(f"{name}.weight", layer.weight), (f"{name}.bias", layer.bias)]
        return named

    def parameters(self):
        return [p for _, p in self.named_parameters()]
