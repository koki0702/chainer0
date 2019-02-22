import contextlib
from chainer0.variable import Variable


class Link(object):

    def __init__(self, **params):
        self._within_init_scope = False
        self._params = set()

        with self.init_scope():
            for name, param in params.items():
                setattr(self, name, param)

    def __call__(self, *args, **kwargs):
        forward = getattr(super(Link, self), '__call__', None)
        if forward is None:
            forward = self.forward
        out = forward(*args, **kwargs)
        return out

    def __setattr__(self, name, value):
        if isinstance(value, Variable) and self._within_init_scope:
            self._params.add(name)
        super().__setattr__(name, value)

    def params(self):
        for name in self._params:
            value = self.__dict__[name]
            if value is not None:
                yield value

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    @contextlib.contextmanager
    def init_scope(self):
        old_flg = self._within_init_scope
        self._within_init_scope = True
        try:
            yield
        finally:
            self._within_init_scope = old_flg


class Chain(Link):

    def __init__(self, **links):
        super().__init__()
        self._children = set()

        with self.init_scope():
            for name, link in links.items():
                setattr(self, name, link)

    def __setattr__(self, name, link):
        if isinstance(link, Link) and self._within_init_scope:
            self._children.add(name)
        super().__setattr__(name, link)

    def params(self):
        for name in self._children:
            link = self.__dict__[name]
            for param in link.params():
                yield param

