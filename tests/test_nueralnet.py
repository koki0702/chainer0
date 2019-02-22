import unittest
import numpy as np
from chainer0 import Function, Variable
from chainer0.functions import matmul, sigmoid, sum, tanh


def sigmoid2(x):
    return 0.5 * (tanh(x) + 1)


class TestNueralnet(unittest.TestCase):

    def test_train(self):
        np.random.seed(0)
        ws = list()
        ws.append(Variable(np.random.rand(2,3)))
        ws.append(Variable(np.random.rand(3,1)))

        data = Variable(np.array([[0,0],[0,1],[1,0],[1,1]]))
        target = Variable(np.array([[0],[1],[0],[1]]))


        expected = [6.210704901174886,1.4434549528818301,0.809644434846779,0.7585458232291216,0.7437140298400363,0.7316218334889659,0.7198939685708763,0.708293823629362,0.6967362454336858,0.6851547179015602]
        log = []

        for i in range(10):
            t = matmul(data, ws[0])
            t = sigmoid2(t)
            pred = matmul(t, ws[1])
            diff = (pred - target)
            loss = sum(diff * diff)
            loss.backward()

            for w in ws:
                w.data -= w.grad * 0.1
                w.cleargrad()
            #print(loss.data)
            log.append(loss.data)

        self.assertTrue(np.allclose(np.array(log), np.array(expected)))
