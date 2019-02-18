import numpy as np

import chainer0
from chainer0 import Link
from chainer0.functions import matmul, broadcast_to


class Linear(Link):
    def __init__(self, in_size, out_size, nobias=False):
        self.nobias = nobias
        W = np.random.rand(in_size, out_size) * np.sqrt(2/in_size)
        if nobias:
            super().__init__(W=W)
        else:
            b = np.zeros(out_size)
            super().__init__(W=W, b=b)

    def __call__(self, x):
        y = matmul(x, self.W)
        if not self.nobias:
            bb = broadcast_to(self.b, y.data.shape)
            y += bb
        return y
