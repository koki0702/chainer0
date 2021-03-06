import numpy as np
from chainer0 import Link
from chainer0 import Variable
from chainer0.functions import embed_id


class EmbedID(Link):

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None):
        super().__init__()

        self.ignore_label = ignore_label
        if initialW is None:
            initialW = np.random.rand(in_size, out_size) - 0.5 / out_size

        with self.init_scope():
            self.W = Variable(initialW)

    def __call__(self, x):
        return embed_id(x, self.W)
