import numpy as np
from chainer0.function import Function
from chainer0.functions.transpose import transpose
from chainer0.functions.reshape import reshape


class MatMul(Function):
    def forward(self, a, b):
        return a.dot(b)

    def backward(self, gy_var):
        a_var, b_var = self.inputs
        ga_var = matmul(gy_var, transpose(b_var))
        gb_var = matmul(transpose(a_var), gy_var)
        return ga_var, gb_var


def as_mat(x, is_left=True):
    if x.data.ndim == 2:
        return x
    shape = (1, x.data.size) if is_left else (x.data.size, 1)
    return reshape(x, shape)


def get_dot_shape(a_shape, b_shape):
    a_ndim, b_ndim = len(a_shape), len(b_shape)

    if (a_ndim, b_ndim) == (2, 2):
        shape = (a_shape[0], b_shape[1])
    elif (a_ndim, b_ndim) == (1, 2):
        shape = (b_shape[1],)
    elif (a_ndim, b_ndim) == (2, 1):
        shape = (a_shape[0],)
    elif (a_ndim, b_ndim) == (1, 1):
        shape = ()
    else:
        raise Exception("Variable's ndim must be 1 or 2")

    return shape


def matmul(a, b):
    is_mat = a.data.ndim == 2 and b.data.ndim == 2
    if not is_mat:
        y_original_shape = get_dot_shape(a.data.shape, b.data.shape)
        a, b = as_mat(a, True), as_mat(b, False)

    f = MatMul()
    y = f(a, b)

    if not is_mat:
        y = reshape(y, y_original_shape)
    return y
