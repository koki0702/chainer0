import unittest
import numpy as np
from chainer0 import Variable, Chain
from chainer0.links import EmbedID, Linear
from chainer0.functions import sigmoid, tanh, mean_squared_error
from chainer0.optimizers import SGD


class TestEmbedModel(unittest.TestCase):

    def test_train(self):
        x = Variable(np.array([1, 2, 1, 2]))
        target = Variable(np.array([[0], [1], [0], [1]]))

        model = Chain(
            embed=EmbedID(5, 3),
            linear=Linear(3, 1),
        )

        def forward(x):
            x = model.embed(x)
            x = tanh(x)
            x = model.linear(x)
            x = sigmoid(x)
            return x

        optimizer = SGD(lr=0.5)
        optimizer.setup(model)

        np.random.seed(0)
        model.embed.W.data = np.random.rand(5, 3)
        model.linear.W.data = np.random.rand(3, 1)

        log = []
        for i in range(10):
            pred = forward(x)
            loss = mean_squared_error(pred, target)

            model.cleargrads()
            loss.backward()
            optimizer.update()
            log.append(loss.data)
            #print(loss.data)

        expected = np.array([0.25458621375800417
                            ,0.24710456626288174
                            ,0.24017425722643587
                            ,0.23364699169761943
                            ,0.22736806682064464
                            ,0.2211879225084124
                            ,0.2149697611450082
                            ,0.20859448689275056
                            ,0.2019642998089552
                            ,0.195005940360243])

        self.assertTrue(np.alltrue(np.array(log) == expected))
