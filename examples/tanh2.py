import numpy as np
import matplotlib.pyplot as plt

import chainer0
from chainer0 import Function, Variable
from chainer0.functions import exp

x_data = np.linspace(-7, 7, 200)
#x_data = np.array([-0.1])
x = Variable(x_data)

x2 = Variable(np.linspace(-7, 7, 200))

def tanh(x):
    #y = exp(-2.0 * x)
    #return (1.0 - y) / (1.0 + y)
    return (1.0 - exp(-x)) / (1.0 + exp(-x))

y = tanh(x)
y.grad = np.ones_like(x_data)
y.backward(enable_double_backprop=True)

vals = [y.data.flatten()]

for i in range(6):
    vals.append(x.grad.flatten())
    dx = x.grad_var
    chainer0.grad(dx)


for i, v in enumerate(vals):
    plt.plot(x_data, vals[i])
plt.show()