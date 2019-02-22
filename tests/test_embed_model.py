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

        expected = np.array([
            0.25458621375800417,
            0.2494970891829434,
            0.24448024091425227,
            0.23946594948721295,
            0.23438097615442757,
            0.2291499591309078,
            0.22369712825782925,
            0.2179486218688263,
            0.2118356986744064,
            0.20529907506021938])

        self.assertTrue(np.alltrue(np.array(log) == expected))
