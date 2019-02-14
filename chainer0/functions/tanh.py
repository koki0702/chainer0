import numpy as np

from chainer0.function import Function


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x[0]),

    def backward(self, grad_vars):
        gy = grad_vars[0]
        x = self.inputs[0]
        #y = self(x)  # y = tanh(x)
        y = self.outputs[0]
        gx = gy * (1 - y * y)
        return gx,


def tanh(x):
    """Elementwise tanh function."""
    f = Tanh()
    return f(x)[0]
