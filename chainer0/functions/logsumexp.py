import numpy as np
import  chainer0
from chainer0 import Function
from chainer0 import Variable
from chainer0.functions.exponential import exp, log
from chainer0.functions.sum import sum


class LogSumExp(Function):

    def __call__(self, *inputs):
        logsumexp(*inputs)
                

def logsumexp(x, axis=None):
    x_max = chainer0.functions.max(x, axis=axis, keepdims=True)
    t = exp(x - x_max)
    y_sum = sum(t, axis=axis)
    return log(y_sum) + x_max.reshape(y_sum.shape)
