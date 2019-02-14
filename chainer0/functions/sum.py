import numpy as np

import chainer0
from chainer0.function import Function


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, inputs):
        x, = inputs
        ret = x.sum(axis=self.axis, keepdims=self.keepdims)
        ret = np.asarray(ret)
        return ret,

    def backward(self, grad_outputs):
        gy = grad_outputs[0]
        x = self.inputs[0]
        gx = chainer0.functions.broadcast_to(gy, x.data.shape)
        return gx,


def sum(x, axis=None, keepdims=False):
    f = Sum(axis, keepdims)
    return f(x)


class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, inputs):
        x, = inputs
        if x.shape == self.shape:
            return x

        ndim = len(self.shape)
        lead = x.ndim - ndim
        lead_axis = tuple(range(lead))
        axis = tuple([i + lead for i, sx in enumerate(self.shape) if sx == 1])
        y = x.sum(lead_axis + axis, keepdims=True)
        if lead > 0:
            y = y.squeeze(lead_axis)
        return y,

    def backward(self, grad_outputs):
        gy = grad_outputs[0]
        x = self.inputs[0]
        gx = chainer0.functions.broadcast_to(gy, x.data.shape)
        return gx,


def sum_to(x, shape):
    f = Sum(shape)
    return f(x)



