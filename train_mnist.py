
import numpy as np
# coding: utf-8

import matplotlib.pyplot as plt

from load_mnist import load_mnist
from util import smooth_curve
from multi_layer_net import MultiLayerNet
from optimizer import *


# 0: Read the MNIST data ==========
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 1000


# 1: Set the experiment ==========
optimizers = {}
optimizers["SGD"] = SGD(lr=0.01)
optimizers["Momentum"] = Momentum(lr=0.01)
optimizers["AdaGrad"] = AdaGrad(lr=0.01)
optimizers["Adam"] = Adam(lr=0.001)

networks = {}   # Dictionary for storing network structures
train_loss = {} # Dictionary for storing loss values

for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100],
        output_size=10,
    )
    train_loss[key] = []


# 2: Start training ==========
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 3: Draw a graph ==========
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}

x = np.arange(max_iterations)

for key in optimizers.keys():
    plt.plot(
        x,
        smooth_curve(train_loss[key]),
        marker=markers[key],
        markevery=100,
        label=key,
    )

plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

test_acc = {key: [] for key in optimizers.keys()}
eval_iters = []

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

    if i % 100 == 0:
        eval_iters.append(i)
        print("=========== iteration:", i, "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            acc = networks[key].accuracy(x_test, t_test)
            test_acc[key].append(acc)
            print(f"{key} | loss: {loss:.4f} | test acc: {acc:.4f}")