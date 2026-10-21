import numpy as np


def gradcheck(fn, inputs, eps=1e-6, rtol=1e-4, atol=1e-5):
    inputs = tuple(inputs)

    if not inputs:
        raise ValueError("At least one input Tensor is required.")

    for x in inputs:
        if not x.requires_grad:
            raise ValueError("All checked inputs must require gradients.")

    def evaluate():
        result = fn(*inputs)

        if not isinstance(result, type(inputs[0])):
            raise TypeError("fn must return a Tensor.")

        if result.size != 1:
            raise ValueError("fn must return a scalar Tensor.")

        return result.item()

    # Analytical gradients: build a fresh graph.
    for x in inputs:
        x.grad = None

    loss = fn(*inputs)
    if loss.size != 1:
        raise ValueError("fn must return a scalar Tensor.")

    loss.backward()

    analytical = []
    for x in inputs:
        if x.grad is None:
            raise AssertionError("Input has no analytical gradient.")
        analytical.append(x.grad.copy())

    # Numerical gradients via central finite differences.
    numerical = []

    for x in inputs:
        original = x.data.copy()
        grad = np.zeros_like(original)

        try:
            for idx in np.ndindex(original.shape):
                x.data[idx] = original[idx] + eps
                plus = evaluate()

                x.data[idx] = original[idx] - eps
                minus = evaluate()

                grad[idx] = (plus - minus) / (2 * eps)
                x.data[idx] = original[idx]
        finally:
            x.data[...] = original

        numerical.append(grad)

    for i, (analytic, numeric) in enumerate(zip(analytical, numerical)):
        np.testing.assert_allclose(
            analytic,
            numeric,
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient mismatch for input {i}",
        )

    return True