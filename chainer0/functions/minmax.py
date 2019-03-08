import numpy as np
import chainer0
from chainer0 import Function


class Max(Function):

    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return np.max(x, axis=self.axis, keepdims=self.keepdims)

    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]

        if self.axis is None:
            axis = range(x.ndim)
        elif isinstance(self.axis, int):
            axis = (self.axis,)
        else:
            axis = self.axis

        # Add broadcastable dimensions to y and gy
        # for each one that was reduced in the forward operation
        shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
        gy = gy.reshape(shape)
        y = y.reshape(shape)
        cond = (x.data == y.data)
        gy = chainer0.functions.broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):

    def forward(self, x):
        return np.min(x, axis=self.axis, keepdims=self.keepdims)


def max(x, axis=None, keepdims=False):
    f = Max(axis, keepdims)
    return f(x)


def min(x, axis=None, keepdims=False):
    f = Min(axis, keepdims)
    return f(x)
