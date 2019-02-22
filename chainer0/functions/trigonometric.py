import numpy as np

from chainer0.function import Function


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0]
        gx = cos(x) * gy
        return gx


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x = self.inputs[0]
        gx = -sin(x) * gy
        return gx


def sin(x):
    """Elementwise sin function."""
    f = Sin()
    return f(x)


def cos(x):
    """Elementwise sin function."""
    f = Cos()
    return f(x)
