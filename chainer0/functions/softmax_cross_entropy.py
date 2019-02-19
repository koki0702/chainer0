import numpy as np
from chainer0 import Function, Variable
from chainer0.functions import exp, sum, log




def _softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x

def _cross_entropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

        # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


class SoftmaxCrossEntropy(Function):

    def forward(self, x, t):
        y = _softmax(x)
        self.y = y

        # 教師ラベルがone-hotベクトルの場合、正解のインデックスに変換
        if t.size == y.size:
            t = t.argmax(axis=1)
        self.t = t

        loss = _cross_entropy(y, t)
        return loss

    def backward(self, dout):
        N = self.y.shape[0]

        dx = self.y.copy()
        dx[np.arange(N), self.t] -= 1
        dx = Variable(dx)

        dx *= dout
        dx = dx / N

        return dx, None


def softmax_cross_entropy(x, t):
    f = SoftmaxCrossEntropy()
    return f(x, t)

'''
def softmax_cross_entropy(x, t):
    tmp = exp(x)
    axis = len(x.data.shape) - 1
    prob = tmp / sum(tmp, axis=axis, keepdims=True)

    t_flatten = t.data.flatten()
    p = prob.data.reshape(len(t_flatten), -1)
    t
    loss =
    return total / N
'''