import numpy as np
from chainer0.variable import Variable


class Optimizer(object):
    def setup(self, link):
        self.target = link

    def update(self):
        for param in self.target.params():
                self.update_one(param)

    def update_one(self, param):
        NotImplementedError()