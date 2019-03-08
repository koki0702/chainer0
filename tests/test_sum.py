import unittest
import numpy as np
from chainer0 import Variable
import chainer0.functions as F
from chainer0.gradient_check import check_backward


class TestSum(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.random.rand(10))
        y = F.sum(x)
        expected = np.sum(x.data)
        self.assertTrue(np.allclose(y.data, expected))

    def test_forward2(self):
        x = Variable(np.random.rand(10,20,30))
        y = F.sum(x, axis=1)
        expected = np.sum(x.data, axis=1)
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward(self):
        x_data = np.random.rand(10)
        f = lambda x: F.sum(x)
        check_backward(f, x_data, testcase=self)

    def test_backward2(self):
        x_data = np.random.rand(10, 10)
        f = lambda x: F.sum(x, axis=1)
        check_backward(f, x_data, testcase=self)

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 20)
        f = lambda x: F.sum(x, axis=(0,2), keepdims=True)
        check_backward(f, x_data, testcase=self)

    def test_backward3(self):
        x_data = np.random.rand(2,3)
        f = lambda x: F.sum(x, axis=1)
        check_backward(f, x_data, testcase=self)


class TestSumTo(unittest.TestCase):

    def test_forward(self):
        x = Variable(np.random.rand(10))
        y = F.sum_to(x, (1,))
        expected = np.sum(x.data)
        self.assertTrue(np.allclose(y.data, expected))

    def test_backward(self):
        x_data = np.random.rand(10)
        f = lambda x: F.sum_to(x, (1,))
        check_backward(f, x_data, testcase=self)

    def test_backward2(self):
        x_data = np.random.rand(10, 10)
        f = lambda x: F.sum_to(x, (10,))
        check_backward(f, x_data, testcase=self)

    def test_backward3(self):
        x_data = np.random.rand(10, 20, 20)
        f = lambda x: F.sum_to(x, (10,))
        check_backward(f, x_data, testcase=self)
