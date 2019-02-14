import numpy as np

from chainer0.function import Function


class Sin(Function):
    def forward(self, x):
        return np.sin(x[0]),

    def backward(self, x, gy):
        gx = np.cos(x[0]) * gy[0]
        return gx,


def sin(x):
    """Elementwise sin function."""
    f = Sin()
    return f(x)[0]
