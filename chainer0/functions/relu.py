import numpy as np

from chainer0.function import Function


class ReLU(Function):
    def forward(self, x):
        return np.maximum(x, 0)

    def backward(self, gy):
        x_var = self.inputs[0]
        gx = np.cos(x[0]) * gy[0]
        return gx,


def relu(x):
    """Rectified Linear Unit function."""
    f = ReLU()
    return f(x)
