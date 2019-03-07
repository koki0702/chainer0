from chainer0.function import Function
from chainer0.functions.reshape import reshape

class Flatten(Function):

    def forward(self, x):
        self.shape = x.shape
        return x.flatten()

    def backward(self, gy):
        return reshape(gy, self.shape)


def flatten(x):
    f = Flatten()
    return f(x)
