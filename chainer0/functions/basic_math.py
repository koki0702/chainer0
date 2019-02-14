import numpy as np
from chainer0.function import Function
from chainer0 import variable
from chainer0 import functions


class Add(Function):
    def forward(self, x):
        y = x[0] + x[1]
        return y,

    def backward(self, gy):
        return gy[0], gy[0]


class Sub(Function):
    def forward(self, x):
        y = x[0] - x[1]
        return y,

    def backward(self, gy):
        return gy[0], -gy[0]


class Mul(Function):
    def forward(self, x):
        y = x[0] * x[1]
        return y,

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy[0]*x1, gy[0]*x0


class Div(Function):
    def forward(self, x):
        y = x[0] / x[1]
        return y,

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy[0] / x1
        gx1 = -gx0 * x0 / x1
        return gx0, gx1


class Neg(Function):
    def forward(self, x):
        return -x[0],

    def backward(self, gy):
        return -gy[0],


class Pow(Function):
    def forward(self, x):
        y = x[0] ** x[1]
        return y,

    def backward(self, grad_vars):
        x0, x1 = self.inputs
        gy = grad_vars[0]

        gx0 = x1 * (x0 ** (x1 - 1)) * gy
        gx1 = functions.log(x0) * (x0 ** x1) * gy
        return gx0, gx1


class Absolute(Function):
    def forward(self, x):
        y = abs(x[0])
        return y,

    def backward(self, gy):
        y = self.output[0]
        sign = variable(np.sign(y.data))
        return sign * gy[0]


def add(self, rhs):
    if not isinstance(rhs, variable.Variable):
        rhs = variable.Variable(np.array(rhs))
    f = Add()
    return f(self, rhs)

def sub(self, rhs):  # lhs - rhs
    if not isinstance(rhs, variable.Variable):
        rhs = variable.Variable(np.array(rhs))
    f = Sub()
    return f(self, rhs)

def rsub(self, rhs):  # rhs - lhs
    if not isinstance(rhs, variable.Variable):
        rhs = variable.Variable(np.array(rhs))
    f = Sub()
    return f(rhs, self)

def mul(self, rhs):
    if not isinstance(rhs, variable.Variable):
        rhs = variable.Variable(np.array(rhs))
    f = Mul()
    return f(self, rhs)

def pow(self, rhs):
    if not isinstance(rhs, variable.Variable):
        rhs = variable.Variable(np.array(rhs))
    f = Pow()
    return f(self, rhs)

def rpow(self, rhs):
    if not isinstance(rhs, variable.Variable):
        rhs = variable.Variable(np.array(rhs))
    f = Pow()
    return f(rhs, self)

def neg(self):
    f = Neg()
    return f(self)


def absolute(self):
    f = Absolute()
    return f(self)


def div(self, rhs):
    if not isinstance(rhs, variable.Variable):
        rhs = variable.Variable(np.array(rhs))
    f = Div()
    return f(self, rhs)

def rdiv(self, rhs):
    if not isinstance(rhs, variable.Variable):
        rhs = variable.Variable(np.array(rhs))
    f = Div()
    return f(rhs, self)

'''
def mul(self, rhs):
    if isinstance(rhs, variable.Variable):
        f = Mul()
        return f(self, rhs)
    f = MulConstant(rhs)
    return f(self)
'''

def install_variable_arithmetics():
    variable.Variable.__neg__ = neg
    variable.Variable.__abs__ = absolute
    variable.Variable.__add__ = add
    variable.Variable.__radd__ = add
    variable.Variable.__sub__ = sub
    variable.Variable.__rsub__ = rsub
    variable.Variable.__mul__ = mul
    variable.Variable.__rmul__ = mul
    variable.Variable.__pow__ = pow
    variable.Variable.__rpow__ = rpow
    variable.Variable.__div__ = div
    variable.Variable.__truediv__ = div
    variable.Variable.__rdiv__ = rdiv
    variable.Variable.__rtruediv__ = rdiv

    '''
    - variable.Variable.__neg__ = neg
    - variable.Variable.__abs__ = absolute
    - variable.Variable.__add__ = add
    - variable.Variable.__radd__ = add
    - variable.Variable.__sub__ = sub
    - variable.Variable.__rsub__ = rsub
    - variable.Variable.__mul__ = mul
    - variable.Variable.__rmul__ = mul
    - variable.Variable.__div__ = div
    - variable.Variable.__truediv__ = div
    - variable.Variable.__rdiv__ = rdiv
    - variable.Variable.__rtruediv__ = rdiv
    variable.Variable.__floordiv__ = floordiv
    variable.Variable.__rfloordiv__ = rfloordiv
    - variable.Variable.__pow__ = pow
    - variable.Variable.__rpow__ = rpow
    variable.Variable.__matmul__ = matmul
    variable.Variable.__rmatmul__ = rmatmul
    '''