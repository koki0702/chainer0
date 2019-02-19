import numpy as np

from chainer0.function import Function


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0]
        gx = np.cos(x) * gy
        return gx,


def sin(x):
    """Elementwise sin function."""
    f = Sin()
    return f(x)
