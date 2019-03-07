import unittest
import numpy as np

import numpy as np
import chainer0
from chainer0.datasets import mnist
import chainer0.links as L
import chainer0.functions as F
import chainer0.distributions as D
from chainer0 import Variable


class TestRegression(unittest.TestCase):

    def test_train(self):
        def _d(*args):
            s = args[0] if len(args) == 1 else args[0] * args[1]
            return np.arange(0, s).reshape(args) * 0.01

        # Network definition
        class MLP(chainer0.Chain):

            def __init__(self, n_in, n_units, n_out):
                super().__init__()
                with self.init_scope():
                    self.l1 = L.Linear(n_in, n_units)
                    self.l2 = L.Linear(n_units, n_units)
                    self.l3 = L.Linear(n_units, n_out)

            def forward(self, x):
                x = F.tanh(self.l1(x))
                x = F.tanh(self.l2(x))
                return self.l3(x)

        def logprob(model, x, t, noise_scale=0.1):
            pred = model(x)
            logp = D.Normal(t, noise_scale).log_prob(pred)
            return -1 * F.sum(logp)

        def build_toy_dataset(n_data=80, noise_std=0.1):
            rs = np.random.RandomState(0)
            inputs = np.concatenate([np.linspace(0, 3, num=int(n_data / 2)),
                                     np.linspace(6, 8, num=int(n_data / 2))])
            targets = np.cos(inputs) + rs.randn(n_data) * noise_std
            inputs = (inputs - 4.0) / 2.0
            inputs = inputs[:, np.newaxis]
            targets = targets[:, np.newaxis] / 2.0
            return inputs, targets


        def init_random_params(model, init_scale=0.1):
            for param in model.params():
                param.data = _d(*param.data.shape)

        model = MLP(1, 4, 1)
        init_random_params(model)

        x, t = build_toy_dataset()
        x, t = Variable(x), Variable(t)

        loss = logprob(model, x, t)
        expected = 399.9346419766375

        self.assertEqual(float(loss.data), float(expected))
