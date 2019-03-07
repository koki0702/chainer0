import numpy as np
from chainer0 import Function
from chainer0.functions.tanh import tanh


class Accuracy(Function):

    def forward(self, y, t):
        pred = y.argmax(axis=1).reshape(t.shape)
        return np.asarray(pred == t).mean()

    def backward(self, gy):
        pass

def accuracy(y, t):
    f = Accuracy()
    return f(y, t)

