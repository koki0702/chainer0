import numpy as np
from chainer0 import Link, Variable
import chainer0.functions as F


class Linear(Link):
    def __init__(self, in_size, out_size=None, nobias=False):
        super().__init__()

        self.nobias = nobias
        if out_size is None:
            in_size, out_size = out_size, in_size
        self.out_size = out_size

        with self.init_scope():
            self.W = Variable(None)
            if in_size is not None:
                self._initialize_params(in_size)
            self.b = None if nobias else Variable(np.zeros(out_size))

    def _initialize_params(self, in_size):
        self.W.data = np.random.rand(in_size, self.out_size) * np.sqrt(2/in_size)

    def __call__(self, x):
        if self.W.data is None:
            in_size = x.shape[1]
            self._initialize_params(in_size)

        y = F.linear(x, self.W, self.b)
        return y
