import numpy as np
from chainer0.function import Function
from chainer0.variable import Variable

class EmbedId(Function):

    def forward(self, x, W):
        return W[x]

    def backward(self, dout):
        x, W = self.inputs
        gW = Variable(np.zeros_like(W.data))
        np.add.at(gW.data, x.data, dout.data)  # TODO: not Varialbe computation
        return None, gW


def embed_id(x, W):
    """Elementwise tanh function."""
    f = EmbedId()
    return f(x, W)
