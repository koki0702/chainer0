import numpy as np
from chainer0.function import Function


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]
        gx = gy * (1 - y * y)
        return gx


def tanh(x):
    """Elementwise tanh function."""
    f = Tanh()
    return f(x)
