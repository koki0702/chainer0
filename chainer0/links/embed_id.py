import numpy as np

import chainer0
from chainer0 import Link
from chainer0.functions import matmul, broadcast_to


class EmbedID(Link):

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None):
        self.ignore_label = ignore_label

        if initialW is None:
            initialW = np.random.rand(in_size, out_size) - 0.5 / out_size
        super().__init__(W=initialW)

    def __call__(self, x):

        y = matmul(x, self.W)
        if not self.nobias:
            bb = broadcast_to(self.b, y.data.shape)
            y += bb
        return y
