#!/usr/bin/env python
"""Fully-connected neural network example using MNIST dataset
This code is a custom loop version of train_mnist.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
import numpy as np
import matplotlib.pyplot as plt

import chainer0
from chainer0.datasets import mnist
import chainer0.links as L
import chainer0.functions as F
import chainer0.distributions as D
from chainer0 import Variable

#from chainer0 import serializers


def _d(*args):
    s = args[0] if len(args) == 1 else args[0] * args[1]
    return np.arange(0,s).reshape(args) * 0.01

# Network definition
class MLP(chainer0.Chain):

    def __init__(self, n_in, n_units, n_out):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_units)
            self.l2 = L.Linear(n_units, n_units)
            self.l3 = L.Linear(n_units, n_out)

    def forward(self, x):
        x = F.tanh(self.l1(x))
        x = F.tanh(self.l2(x))
        return self.l3(x)

def log_gaussian(model, scale):
    t = 0
    normal = D.Normal(0, scale)
    for p in model.params():
        t += F.sum(normal.log_prob(p))
    return -1 * t

def logprob(model, x, t, noise_scale=0.1):
    pred = model(x)
    logp = D.Normal(t, noise_scale).log_prob(pred)
    return -1 * F.sum(logp)

def build_toy_dataset(n_data=80, noise_std=0.1):
    rs = np.random.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 3, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 2.0
    inputs  = inputs[:, np.newaxis]
    targets = targets[:, np.newaxis] / 2.0
    return inputs, targets

def plot_nn(model, ax, inputs, targets):
    # Plot data and functions.
    plt.cla()
    ax.plot(inputs.ravel(), targets.ravel(), 'bx', ms=12)
    plot_inputs = np.reshape(np.linspace(-7, 7, num=300), (300, 1))

    outputs = model(Variable(plot_inputs))
    ax.plot(plot_inputs, outputs.data, 'r', lw=3)
    ax.set_ylim([-1, 1])
    plt.draw()
    plt.pause(1.0 / 60.0)

def init_random_params(model, init_scale=0.1):
    for param in model.params():
        param.data = _d(*param.data.shape)


def main():
    model = MLP(1, 4, 1)
    init_random_params(model)

    optimizer = chainer0.optimizers.SGD(lr=0.0001)  # chainer0.optimizers.Adam()
    optimizer.setup(model)
    x, t = build_toy_dataset()
    x, t = Variable(x), Variable(t)

    max_iter = 1000
    weight_prior_variance = 10.0
    # Set up figure.
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    for i in range(max_iter):
        model.cleargrads()
        loss = logprob(model, x, t)
        loss += log_gaussian(model, weight_prior_variance)
        loss.backward()
        optimizer.update()

        if i % 100 == 0:
            plot_nn(model, ax, x.data, t.data)
            print("Iteratoin {} log likelihood {}".format(i, loss.data))



if __name__ == '__main__':
    main()