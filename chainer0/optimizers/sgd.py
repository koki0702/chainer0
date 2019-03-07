from chainer0.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad