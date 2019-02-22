from chainer0.function import Function


class Reshape(Function):
    def __init__(self, shape):
        self.shape =shape

    def forward(self, x):
        return x.reshape(self.shape)

    def backward(self, gy):
        x = self.inputs[0]
        return reshape(gy, x.shape)


def reshape(x, shape):
    f = Reshape(shape)
    return f(x)
