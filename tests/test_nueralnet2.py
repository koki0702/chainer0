import unittest
import numpy as np
from chainer0 import Variable, Chain
from chainer0.links import Linear
from chainer0.optimizers import SGD
from chainer0.functions import sigmoid, mean_squared_error


class TestNueralnet2(unittest.TestCase):

    def test_train(self):
            data = Variable(np.array([[0,0],[0,1],[1,0],[1,1]]))
            target = Variable(np.array([[0],[1],[0],[1]]))

            model = Chain(
                f1=Linear(2, 3, nobias=True),
                f2=Linear(3,1, nobias=True),
            )

            np.random.seed(0)

            model.f1.W.data = np.random.rand(2,3)
            model.f2.W.data = np.random.rand(3,1)

            expected = [1.0820313915580297,0.6937790311924391,0.4807580742799925,0.36192715863503955,0.29491127763318664,0.25681463756748657,0.23500603629345754,0.22242443947250173,0.21508930205325136,0.21074454544426313]
            log = []
            def forward(x):
                h = sigmoid(model.f1(x))
                return model.f2(h)

            optimizer = SGD(lr=0.1)
            optimizer.setup(model)

            for i in range(10):
                y = forward(data)
                loss = mean_squared_error(y, target)
                model.cleargrads()
                loss.backward()
                optimizer.update()
                #print(loss.data)
                log.append(loss.data)

            self.assertTrue(np.alltrue(np.array(log) == np.array(expected)))
