import numpy as np
from chainer0.function import Function
from chainer0.functions.matmul import matmul
from chainer0.functions.broadcast import broadcast_to


class Linear(Function):

    def __call__(self, *inputs):
        return linear(*inputs)


def linear(x, W, b=None):
    y = matmul(x, W)
    if b is not None:
        bb = broadcast_to(b, y.data.shape)
        y += bb
    return y
