import unittest
import numpy as np
from chainer0 import Variable
import chainer0.functions as F
from chainer0.gradient_check import check_backward


class TestTanh(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.random.rand(1))
        y = F.tanh(x)
        expected = np.tanh(x.data)
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward(self):
        x_data = np.random.rand(1)
        check_backward(F.tanh, x_data, testcase=self)


    def test_backward2(self):
        x_data = np.random.rand(10,10)
        y_grad = np.ones_like(x_data)
        check_backward(F.tanh, x_data, y_grad, testcase=self)