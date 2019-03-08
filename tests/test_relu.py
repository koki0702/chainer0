import unittest
import numpy as np
from chainer0 import Variable
import chainer0.functions as F
from chainer0.gradient_check import check_backward


class TestRelu(unittest.TestCase):

    def test_forward(self):
        x_data = np.array([-1, -2, 3, 4])
        x = Variable(x_data)
        y = F.relu(x)
        x_data[x_data < 0] = 0
        expected = x_data
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward(self):
        x_data = np.random.rand(5,5,5)
        check_backward(F.relu, x_data, testcase=self)
    """
    def test_backward2(self):
        x_data = np.random.rand(10,10)
        y_grad = np.ones_like(x_data)
        check_backward(F.tanh, x_data, y_grad, testcase=self)
    """