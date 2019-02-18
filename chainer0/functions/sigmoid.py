from chainer0.functions.tanh import tanh


def sigmoid(x):
    return 0.5 * (tanh(x) + 1)
