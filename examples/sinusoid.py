import numpy as np
import matplotlib.pyplot as plt
import chainer0
import chainer0.functions as F
from chainer0 import Function, Variable, grad


def fun(x):
    return F.sin(x)

x = Variable(np.linspace(-10, 10, 100))
y = fun(x)
gx, = chainer0.grad([y], [x], enable_double_backprop=True)
ggx, = chainer0.grad([gx], [x], enable_double_backprop=True)
plt.plot(x.data, y.data)
plt.plot(x.data, gx.data)
plt.plot(x.data, ggx.data)
plt.show()


# Taylor approximation to sin function
def fun(x):
    currterm = x
    ans = currterm
    for i in range(1000):
        #print(i, end=' ')
        currterm = - currterm * x ** 2 / ((2 * i + 3) * (2 * i + 2))
        ans = ans + currterm
        if np.abs(currterm.data) < 0.2: break  # (Very generous tolerance!)

    return ans

x_list = np.linspace(-10, 10, 100)
y_list, gx_list, ggx_list = [], [], []

for x in x_list:
    x = Variable(x)
    y = fun(x)
    gx, = chainer0.grad([y], [x], enable_double_backprop=True)
    ggx, = chainer0.grad([gx], [x], enable_double_backprop=True)

    y_list.append(y.data)
    gx_list.append(gx.data)
    ggx_list.append(ggx.data)


plt.plot(x_list, y_list)
plt.plot(x_list, gx_list)
plt.plot(x_list, ggx_list)
plt.show()

'''
def taylor_sine(x):  # Taylor approximation to sine function
    ans = currterm = x
    i = 0
    while abs(currterm).data > 0.001:
        currterm = -currterm * x ** 2 / ((2 * i + 3) * (2 * i + 2))
        ans = ans + currterm
        i += 1
    return ans

x = Variable(np.array([np.pi]))
y = taylor_sine(x)
y.backward()
print('Gradient of sin(pi) is',  x.grad[0])
'''