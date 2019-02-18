import numpy as np

from chainer0.functions import sum


def mean_squared_error(x0, x1):
    diff = x0 - x1
    total = sum(diff * diff)
    N = x0.data.shape[0]
    return total / N
