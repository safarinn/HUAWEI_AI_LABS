import numpy as np


def numerical_gradient(f, x):
    """Compute numerical gradient of function f at x.

    Args:
        f: function that takes x and returns scalar loss.
        x: np.ndarray, point to evaluate gradient at.
    """
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]

        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()

    return grad

