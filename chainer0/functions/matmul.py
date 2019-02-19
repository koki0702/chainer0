import numpy as np

from chainer0.function import Function
from chainer0.functions import transpose


def as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class MatMul(Function):
    def forward(self, a, b):
        #a, b = as_mat(a), as_mat(b)
        return a.dot(b)

    def backward(self, gy_var):
        a_var, b_var = self.inputs
        ga_var = matmul(gy_var, transpose(b_var))
        gb_var = matmul(transpose(a_var), gy_var)
        return ga_var, gb_var


def matmul(a, b):
    f = MatMul()
    return f(a, b)
