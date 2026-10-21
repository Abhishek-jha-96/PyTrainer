## Custom Autodiff Engine

Implemented a lightweight, NumPy-based automatic differentiation engine from scratch, covering the core building blocks required for gradient-based computation and optimization.

### Features

- **Tensor Abstraction:** Implemented a `Tensor` class to store numerical data, track gradients, and maintain computational graph relationships.
- **Topological Backward Traversal:** Built reverse-mode automatic differentiation using topological ordering to propagate gradients through the computational graph.
- **Broadcasting-Aware Gradient Reduction:** Added gradient reduction to correctly handle NumPy broadcasting rules during backpropagation.
- **Element-wise Arithmetic:** Implemented differentiable arithmetic operations, including addition, subtraction, multiplication, division, and power.
- **Matrix Multiplication:** Added 2D matrix multiplication with gradients propagated to both input tensors.
- **Reduction Operations:** Implemented sum and mean reductions with support for axes and dimension preservation.
- **Nonlinear Activations:** Added ReLU and tanh activations with their corresponding backward functions.
- **Stable Softmax Cross-Entropy:** Implemented numerically stable softmax cross-entropy loss with batch-mean normalization and support for class-index and one-hot targets.
- **Automated Gradient Checking:** Built a central finite-difference gradient checker to compare analytical gradients against numerical approximations, validating individual operations, broadcasting, and combined computational graphs.
