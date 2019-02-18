import numpy as np

from chainer0.function import Function
from chainer0.functions import transpose


class Linear(Function):
    def forward(self, inputs):
        x, W = inputs[0], inputs[1]
        y = x.dot(W)
        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y

    def backward(self, grad_vars):
        gy_var = grad_vars[0]
        a_var, b_var = self.inputs
        ga_var = matmul(gy_var, transpose(b_var))
        gb_var = matmul(transpose(a_var), gy_var)
        return ga_var, gb_var


def matmul(a, b):
    f = MatMul()
    return f(a, b)[0]
