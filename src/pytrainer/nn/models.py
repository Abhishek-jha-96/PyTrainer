from pytrainer.autodiff import ops
from pytrainer.nn.layers import Linear


class MLP:
    """2 -> 32 -> 16 -> 2 multilayer perceptron with ReLU activations.

    The final layer returns raw logits (no softmax). Normalization is
    handled inside the stable softmax cross-entropy loss.
    """

    def __init__(self, rng):
        self.fc1 = Linear(2, 32, rng, init="he_normal")
        self.fc2 = Linear(32, 16, rng, init="he_normal")
        self.fc3 = Linear(16, 2, rng, init="xavier_uniform")

    def __call__(self, x):
        x = ops.relu(self.fc1(x))
        x = ops.relu(self.fc2(x))
        x = self.fc3(x)  # raw logits
        return x

    def parameters(self):
        return (
            self.fc1.parameters()
            + self.fc2.parameters()
            + self.fc3.parameters()
        )