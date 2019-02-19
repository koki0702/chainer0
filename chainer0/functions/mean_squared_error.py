import numpy as np
from chainer0 import Function
from chainer0.functions import sum


def MeanSquaredError(Function):

    def __call__(*inputs):
        x, t = inputs
        mean_squared_error(x, t)


def mean_squared_error(x, t):
    diff = x - t
    total = sum(diff * diff)
    N = x.data.shape[0]
    return total / N
