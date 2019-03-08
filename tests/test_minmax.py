import unittest
import numpy as np
from chainer0 import Variable
import chainer0.functions as F
from chainer0.gradient_check import check_backward


class TestMinMax(unittest.TestCase):

    def test_forward(self):
        x_data =np.random.rand(10)
        x = Variable(x_data)
        y = F.max(x)
        expected = np.max(x_data)

        self.assertTrue(y.data == expected)


    def test_forward2(self):
        x_data =np.random.rand(10, 10)
        x = Variable(x_data)
        y = F.max(x)
        expected = np.max(x_data)
        self.assertTrue(y.data == expected)


    def test_backward(self):
        x_data = np.random.rand(10)
        check_backward(F.max, x_data, testcase=self, verbose=True)

    def test_backward2(self):
        x_data = np.random.rand(5,5)
        check_backward(F.max, x_data, testcase=self)

    def test_backward3(self):
        x_data = np.random.rand(5,5)
        f = lambda x:F.max(x, axis=1, keepdims=True)
        check_backward(f, x_data, testcase=self)

    def test_backward4(self):
        x_data = np.random.rand(5,5,5)
        f = lambda x:F.max(x, axis=None, keepdims=True)
        check_backward(f, x_data, testcase=self)
    """
    def test_backward2(self):
        x_data = np.random.rand(10,10)
        y_grad = np.ones_like(x_data)
        check_backward(F.tanh, x_data, y_grad, testcase=self)
    """