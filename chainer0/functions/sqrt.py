import numpy as np
from chainer0.function import Function


class Sqrt(Function):

    def forward(self, x):
        return np.sqrt(x)

    def backward(self, gy):
        x = self.outputs[0]
        return gy / (x * 2.0)


def sqrt(x):
    f = Sqrt()
    return f(x)
