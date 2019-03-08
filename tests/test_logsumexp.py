import unittest
import numpy as np
from chainer0 import Variable
import chainer0.functions as F
from chainer0.gradient_check import check_backward


def simple_logsumexp(x, axis=None):
    return np.log(np.exp(x).sum(axis=axis))
    
class TestLogSumExp(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.array([1,2,3,4]))
        y = F.logsumexp(x)
        expected = simple_logsumexp(x.data)
        res = np.allclose(y.data, expected)

        self.assertTrue(res)

    def test_forward2(self):
        x = Variable(np.random.randn(2,3))
        y = F.logsumexp(x, axis=1)
        expected = simple_logsumexp(x.data, axis=1)
        res = np.allclose(y.data, expected)
        self.assertTrue(res)


    def test_backward(self):
        x_data = np.random.rand(10)
        check_backward(F.logsumexp, x_data, testcase=self)


    def test_backward2(self):
        x_data = np.random.rand(30,10)
        f = lambda x:F.logsumexp(x)
        check_backward(f, x_data, testcase=self)


    def test_backward2(self):
        x_data = np.random.rand(3,10)
        f = lambda x:F.logsumexp(x, axis=1)
        check_backward(f, x_data, testcase=self)