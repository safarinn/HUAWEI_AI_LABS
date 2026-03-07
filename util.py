import numpy as np


def smooth_curve(x):
    """Smooth a 1D loss curve using a simple moving average."""
    x = np.array(x, dtype=np.float32)
    window_len = 11

    if x.size < window_len:
        return x

    window = np.ones(window_len) / window_len
    return np.convolve(x, window, mode="valid")

