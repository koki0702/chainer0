import numpy as np
import matplotlib.pyplot as plt

import chainer0
from chainer0 import Function, Variable
from chainer0.functions import tanh

x_data = np.linspace(-7, 7, 200)
#x_data = np.array([-0.1])
x = Variable(x_data)

#def sigmoid(x):
#    return 0.5 * (tanh(x / 2.) + 1)
y = tanh(x)
y.grad = np.ones_like(x_data)
y.backward(enable_double_backprop=True)


vals = [y.data.flatten()]

for i in range(1):
    vals.append(x.grad.flatten())
    dx = x.grad_var
    chainer0.grad(dx)


for i, v in enumerate(vals):
    plt.plot(x_data, vals[i])
plt.show()