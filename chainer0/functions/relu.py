import numpy as np

from chainer0.function import Function


class ReLU(Function):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, gy):
        y = self.outputs[0]
        gx = gy * (y.data > 0)
        return gx


def relu(x):
    """Rectified Linear Unit function."""
    f = ReLU()
    return f(x)
