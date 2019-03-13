import unittest
import numpy as np

from chainer0 import Variable

import chainer0.functions as F
from chainer0.gradient_check import check_backward
from chainer0.distributions.normal import Normal

class TestNormalDistributioin(unittest.TestCase):

    def test_forward(self):
        y = Normal(np.array(0.0), np.array(1.0)).prob(Variable(np.array(0)))
        expected = 0.3989422804014327

        self.assertEqual(y.data, expected)

    def test_forward2(self):
        y = Normal(np.array(0.0), np.array(1.0)).prob(Variable(np.array(0.5)))
        expected = 0.3520653267642995


        self.assertEqual(y.data, expected)

    """
    def test_backward_prob(self):
        f = Normal(0.0, 1.0).prob
        x_data = np.random.rand(1)
        check_backward(f, x_data, testcase=self, verbose=True)
    """

    def test_backward_prob2(self):
        f = Normal(np.array(0.0), np.array(1.0)).prob
        x_data = np.random.rand(10)
        check_backward(f, x_data, testcase=self, verbose=True)

