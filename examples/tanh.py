import numpy as np
import matplotlib.pyplot as plt
import chainer0
from chainer0 import Function, Variable
import chainer0.functions as F


x_data = np.linspace(-7, 7, 200)
x = Variable(x_data)
y = F.tanh(x)

y.backward(enable_double_backprop=True)
vals = [y.data.flatten()]


for i in range(4):
    vals.append(x.grad.flatten())
    dx = x.grad_var
    x.cleargrad()
    dx.backward(enable_double_backprop=True)

'''
for i in range(4):
    vals.append(x.grad.flatten())
    dx = x.grad_var
    chainer0.grad(dx)
'''

for i, v in enumerate(vals):
    plt.plot(x_data, vals[i])
plt.show()

