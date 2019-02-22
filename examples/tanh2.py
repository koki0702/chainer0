import numpy as np
import matplotlib.pyplot as plt
import chainer0
from chainer0 import Function, Variable
import chainer0.functions as F


x_data = np.linspace(-7, 7, 200)
x = Variable(x_data)
y = F.tanh(x)

gx, = chainer0.grad([y], [x], enable_double_backprop=True)
vals = [y.data.flatten()]


for i in range(4):
    vals.append(gx.data.flatten())
    gx, = chainer0.grad([gx], [x], enable_double_backprop=True)


for i, v in enumerate(vals):
    plt.plot(x_data, vals[i])
plt.show()

