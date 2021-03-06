import numpy as np
import heapq

import chainer0
from chainer0 import configuration


class Variable(object):

    def __init__(self, data, grad_var=None, name=None):
        self.data = data
        self.rank = 0
        self.grad_var = grad_var
        self.creator = None
        self.name = name

    def __repr__(self):
        return np.array2string(self.data)

    @property
    def grad(self):
        if self.grad_var is None:
            return None
        return self.grad_var.data

    @grad.setter
    def grad(self, g):
        if isinstance(g, Variable):
            self.grad_var = g
        elif g is None:
            self.grad_var = None
        else:
            self.grad_var = Variable(g)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, gen_func):
        self.creator = gen_func
        self.rank = gen_func.rank + 1

    def backward(self, enable_double_backprop=True):
        if self.creator is None:
            return
        #if self.data.size == 1 and self.grad is None:  # Loss variable
        if self.grad is None:
            self.grad_var = Variable(np.ones_like(self.data))

        cand_funcs = []
        seen_set = set()

        def add_cand(cand):
            if cand not in seen_set:
                heapq.heappush(cand_funcs, (-cand.rank, len(seen_set), cand))
                seen_set.add(cand)

        add_cand(self.creator)

        while cand_funcs:
            _, _, func = heapq.heappop(cand_funcs)

            out_grad_var = [y.grad_var for y in func.outputs]
            for y in func.outputs:
                y.grad_var = None#Variable(y.grad_var.data)

            with configuration.using_config('enable_backprop', enable_double_backprop):
                gxs = func.backward(*out_grad_var)

            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(func.inputs, gxs):
                if x.grad_var is None:
                    x.grad_var = gx
                else:
                    x.grad_var = x.grad_var + gx

                if x.creator is not None:
                    add_cand(x.creator)

    def cleargrad(self):
        self.grad_var = None

    def unchain(self):
        self.creator = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return chainer0.functions.reshape(self, shape)

#    def __pow__(self, power, modulo=None):
