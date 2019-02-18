import numpy as np
from chainer0.variable import Variable


class Link(object):

    def __init__(self, **params):
        for name, param in params.items():
            setattr(self, name, Variable(param))

    def params(self):
        for param in self.__dict__.values():
            if isinstance(param, Variable):
                yield param

    def cleargrads(self):
        for param in self.params():
            if isinstance(param, Variable):
                param.cleargrad()

    def init_scope(self):
        pass  # dummy function


class Chain(Link):
    def __init__(self, **links):
        super().__init__()
        for name, link in links.items():
            self.__dict__[name] = link

    def params(self):
        for link in self.__dict__.values():
            if isinstance(link, Link):
                for param in link.params():
                    yield param
