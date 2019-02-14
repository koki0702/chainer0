import numpy as np

from chainer0 import Function, Variable


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
print('Gradient of sin(pi) is',  x.grad)