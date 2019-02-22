import numpy as np
from chainer0.variable import Variable
import chainer0.functions as F


def check_backward(func, inputs_data, y_grad=None, testcase=None, eps=0.001,
                   atol=1e-5, rtol=1e-4, verbose=False):
    inputs_data = _as_tuple(inputs_data)

    def f(inputs):
        inputs = _as_tuple(inputs)
        inputs = [Variable(x) for x in inputs]
        y = func(*inputs)
        return y.data

    num_grads = numerical_grad(f, inputs_data, y_grad, eps)
    inputs = [Variable(x) for x in inputs_data]
    y = func(*inputs)
    if y_grad is not None:
        y.grad = y_grad
    y.backward()
    bp_grads = [x.grad for x in inputs]

    for num_grad, bp_grad in zip(num_grads, bp_grads):
        res = np.allclose(num_grad, bp_grad, atol=atol, rtol=rtol)

        if verbose:
            print(num_grad - bp_grad)

        if testcase is not None:
            testcase.assertTrue(res)

    return True


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def numerical_grad(f, inputs, grad_output=None, eps=0.001):
    h = eps
    inputs = _as_tuple(inputs)
    grads = [np.zeros_like(x) for x in inputs]

    for x, grad in zip(inputs, grads):
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            tmp_val = x[idx]
            x[idx] = float(tmp_val) + h
            fxh1 = f(*inputs)  # f(x+h)

            x[idx] = tmp_val - h
            fxh2 = f(*inputs)  # f(x-h)

            if grad_output is None:
                diff = fxh1 - fxh2
            else:
                diff = ((fxh1 - fxh2) * grad_output).sum()
            grad[idx] = diff / (2 * h)

            x[idx] = tmp_val  # 値を元に戻す
            it.iternext()

    return _as_tuple(grads)


def _get_item(x):
    slice = 1
    return F.get_item(x, slice)

x = np.random.randn(2,3)
y = _get_item(x)
print(x)
print(y)

res= check_backward(_get_item, np.random.randn(20,30), np.ones_like(y))
print(res)
