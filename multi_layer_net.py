import numpy as np
from collections import OrderedDict

from layers import Affine, Relu, Sigmoid, SoftmaxWithLoss
from gradient import numerical_gradient


class MultiLayerNet:
    """Fully-connected multi-layer neural network.

    Parameters
    ----------
    input_size: input size (784 for MNIST)
    hidden_size_list: list with the number of neurons per hidden layer
                      (for example, [100, 100, 100, 100])
    output_size: output size (10 for MNIST)
    activation: 'relu' or 'sigmoid'
    weight_init_std: standard deviation or strategy for the weight initialization.
        If 'relu' or 'he', use He initialization.
        If 'sigmoid' or 'xavier', use Xavier initialization.
    weight_decay_lambda: strength of L2 weight decay.
    """

    def __init__(
        self,
        input_size,
        hidden_size_list,
        output_size,
        activation="relu",
        weight_init_std="relu",
        weight_decay_lambda=0,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        # Initialize weights
        self.__init_weight(weight_init_std)

        # Generate layers
        activation_layer = {"sigmoid": Sigmoid, "relu": Relu}
        self.layers = OrderedDict()
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers["Affine" + str(idx)] = Affine(
                self.params["W" + str(idx)], self.params["b" + str(idx)]
            )
            self.layers["Activation_function" + str(idx)] = activation_layer[
                activation
            ]()

        idx = self.hidden_layer_num + 1
        self.layers["Affine" + str(idx)] = Affine(
            self.params["W" + str(idx)], self.params["b" + str(idx)]
        )

        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std):
        """Set initial weights."""
        all_size_list = (
            [self.input_size] + self.hidden_size_list + [self.output_size]
        )
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ("relu", "he"):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ("sigmoid", "xavier"):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            self.params["W" + str(idx)] = scale * np.random.randn(
                all_size_list[idx - 1], all_size_list[idx]
            )
            self.params["b" + str(idx)] = np.zeros(all_size_list[idx])

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """Compute loss with optional L2 weight decay."""
        y = self.predict(x)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params["W" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W ** 2)

        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        """Gradient calculation (numerical differentiation)."""

        def loss_W(_):
            return self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = numerical_gradient(
                loss_W, self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = numerical_gradient(
                loss_W, self.params["b" + str(idx)]
            )

        return grads

    def gradient(self, x, t):
        """Gradient calculation by backpropagation."""
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Set gradients
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            affine_layer = self.layers["Affine" + str(idx)]
            grads["W" + str(idx)] = (
                affine_layer.dW
                + self.weight_decay_lambda * affine_layer.W
            )
            grads["b" + str(idx)] = affine_layer.db

        return grads

