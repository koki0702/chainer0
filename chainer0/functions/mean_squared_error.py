import numpy as np
from chainer0.function import Function
from chainer0.functions import sum


def MeanSquaredError(Function):

    def __call__(self, *inputs):
        x, t = inputs
        return mean_squared_error(x, t)


def mean_squared_error(x, t):
    diff = x - t
    total = sum(diff * diff)
    N = x.data.shape[0]
    return total / N
