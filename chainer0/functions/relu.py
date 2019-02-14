import numpy as np

from chainer0.function import Function


class ReLU(Function):
    def forward(self, x):
        self.y = np.maximum(x[0], 0)
        return self.y,

    def backward(self, x, gy):
        gx = np.cos(x[0]) * gy[0]
        return gx,


def relu(x):
    """Rectified Linear Unit function."""
    f = ReLU()
    return f(x)[0]
