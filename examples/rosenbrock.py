import numpy as np
import chainer0.functions as F
from chainer0 import Variable
from scipy.optimize import minimize


def rosenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2


def value_and_grad(f):

    def _f(x):
        x = Variable(x)
        y = f(x)
        y.backward()
        return y.data, x.grad
    return _f


rosenbrock_with_grad = value_and_grad(rosenbrock)
# Optimize using conjugate gradients.
result = minimize(rosenbrock_with_grad, x0=np.array([0.0, 0.0]), jac=True, method='CG')
print("Found minimum at {0}".format(result.x))
