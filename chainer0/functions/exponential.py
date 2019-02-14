import numpy as np

from chainer0.function import Function


class Exp(Function):
    def forward(self, x):
        return np.exp(x[0]),

    def backward(self, grad_vars):
        y = self.outputs[0]
        gy = grad_vars[0]
        return y * gy,


class Log(Function):
    def forward(self, x):
        return np.log(x[0]),

    def backward(self, grad_vars):
        x_var = self.inputs[0]
        gx = grad_vars[0] / x_var
        return gx,


def exp(x):
    return Exp()(x)


def log(x):
    f = Log()
    return f(x)
