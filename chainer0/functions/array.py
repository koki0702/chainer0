import numpy as np
from chainer0.function import Function, Variable


class GetItem(Function):
    def __init__(self, slices):
        if isinstance(slices, list):
            if all([isinstance(s, int) for s in slices]):
                slices = slices,
            slices = tuple(slices)
        elif not isinstance(slices, tuple):
            slices = slices,

        self.slices = slices

    def forward(self, x):
        return x[self.slices]

    def backward(self, gy):
        x = self.inputs[0]
        gx = Variable(np.zeros_like(x.data))
        np.add.at(gx.data, self.slices, gy.data) # TODO: not Varialbe computation
        return gx


def get_item(x, slices):
    f = GetItem(slices)
    return f(x)


def install_variable_get_item():
    Variable.__getitem__ = get_item
