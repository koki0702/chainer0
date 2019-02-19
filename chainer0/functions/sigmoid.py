import numpy as np
from chainer0 import Function
from chainer0.functions.tanh import tanh


class Sigmoid(Function):

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, gy):
        y = self.outputs[0]
        gx = y * (1 - y) * gy
        return gx

def sigmoid(x):
    f = Sigmoid()
    return f(x)

def sigmoid2(x):
    return 0.5 * (tanh(x) + 1)
