import numpy as np

import chainer0
from chainer0.function import Function


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = tuple(shape)

    def forward(self, x):
        return np.broadcast_to(x, self.shape)

    def backward(self, gy):
        x = self.inputs[0]
        return chainer0.functions.sum_to(gy, x.data.shape),


def broadcast_to(x, shape):
    f = BroadcastTo(shape)
    return f(x)
