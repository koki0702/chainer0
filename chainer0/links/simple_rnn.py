import numpy as np
from chainer0 import Chain, Variable
import chainer0.functions as F
from chainer0.links.linear import Linear


class SimpleRNN(Chain):

    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size

        with self.init_scope():
            self.l1 = Linear(in_size, hidden_size)
            self.l2 = Linear(hidden_size, hidden_size)
        self.reset_state()

    def forward(self, x):
        if self.h is None:
            batch_size = x.data.shape[0]
            self.h = Variable(np.zeros((batch_size, self.hidden_size)))

        a = self.l1(x) + self.l2(self.h)
        self.h = F.tanh(a)
        return self.h

    def set_sate(self, h):
        self.h = h

    def reset_state(self):
        self.h = None
