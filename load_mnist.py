import os
import numpy as np
from urllib.request import urlretrieve


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_FILENAME = "mnist.npz"


def _get_mnist_path() -> str:
    """Return local path to the MNIST .npz file, downloading it if needed."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, MNIST_FILENAME)

    if not os.path.exists(path):
        os.makedirs(base_dir, exist_ok=True)
        print(f"Downloading MNIST dataset to {path} ...")
        urlretrieve(MNIST_URL, path)
        print("Download finished.")

    return path


def load_mnist(normalize: bool = True):
    """Load MNIST dataset.

    Returns:
        (x_train, t_train), (x_test, t_test)
        - x_*: np.ndarray of shape (N, 784), dtype float32 if normalize else uint8
        - t_*: np.ndarray of shape (N,), dtype uint8 (labels 0-9)
    """
    path = _get_mnist_path()

    with np.load(path) as data:
        x_train = data["x_train"]
        t_train = data["y_train"]
        x_test = data["x_test"]
        t_test = data["y_test"]

    # Flatten images to 784-dim vectors
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)

    if normalize:
        x_train = x_train.astype(np.float32) / 255.0
        x_test = x_test.astype(np.float32) / 255.0

    return (x_train, t_train), (x_test, t_test)

