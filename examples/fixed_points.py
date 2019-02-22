import numpy as np
import chainer0
import chainer0.functions as F
from chainer0 import Variable


def fixed_point(f, a, x0, distance, tol):
    _f = f(a)
    x, x_prev = _f(x0), x0

    while distance(x, x_prev) > tol:
        x, x_prev = _f(x), x

    return x

def newton_sqrt_iter(a):
    return lambda x: 0.5 * (x + a / x)

def grad_descent_sqrt_iter(a):
    return lambda x: x - 0.05 * (x**2 - a)

def sqrt(a, guess=10.):
    return fixed_point(newton_sqrt_iter, a, guess, distance, 1e-4)
    #return fixed_point(grad_descent_sqrt_iter, a, guess, distance, 1e-4)

def distance(x, y):
    x_data = x.data if isinstance(x, Variable) else x
    y_data = y.data if isinstance(y, Variable) else y
    return np.abs(x_data - y_data)



x = Variable(np.array(2.))
y = F.sqrt(x)
gy, = chainer0.grad([y], [x])
ggy, = chainer0.grad([gy], [x])


x2 = Variable(np.array(2.))
y2 = sqrt(x2)
gy2, = chainer0.grad([y2], [x2])
ggy2, = chainer0.grad([gy2], [x2])

print(y)
print(y2)
print()
print(gy)
print(gy2)
print()
print(ggy)
print(ggy2)

'''
print(grad(np.sqrt)(2.))
print(grad(sqrt)(2.))
print()
print(grad(grad(np.sqrt))(2.))
print(grad(grad(sqrt))(2.))
print()
'''