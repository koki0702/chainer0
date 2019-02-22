import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import Variable
import chainer.functions as F

x_data = np.linspace(-7, 7, 200)
x = Variable(x_data)

y = F.tanh(x)
gx, = chainer.grad([y], [x], enable_double_backprop=True)
vals = [y.data.flatten()]
vals.append(gx.data.flatten())

ggx, = chainer.grad([gx], [x])
vals.append(ggx.data.flatten())

gggx, = chainer.grad([ggx], [x])
vals.append(gggx.data.flatten())

'''
for i in range(4):
    print(type(gx))
    vals.append(gx.data.flatten())
    ggx, = chainer.grad([gx], [x])
    gx = ggx
'''

'''

y = F.tanh(x)
y.grad = np.ones_like(y.data)
y.backward(enable_double_backprop=True)
vals = [y.data.flatten()]

for i in range(4):
    vals.append(x.grad.flatten())
    gx = x.grad_var
    x.cleargrad()
    gx.grad = np.ones_like(y.data)
    gx.backward( enable_double_backprop=True)
    #print(id(x.grad_var))
'''

for i, v in enumerate(vals):
    plt.plot(x_data, vals[i])
plt.show()
