import numpy as np

from chainer0 import Function, Variable
from chainer0.functions.basic_math import Add, Mul

x = Variable(np.array([2.0]))
y = x ** 2 + x + 1.0

y.backward(enable_double_backprop=True)
dx = x.grad_var

print('y', y.data)
print('dx', x.grad)

x.cleargrad()
dx.backward()
print('ddx', x.grad)

dx = x.grad_var
x.cleargrad()
dx.backward()
print('dddx', x.grad)


xx = Variable(np.random.rand(10))

y = 1 + xx
print(xx.data)
print(y.data)
#x.cleargrad()