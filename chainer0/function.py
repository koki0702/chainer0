import numpy as np

from chainer0.variable import Variable
from chainer0 import configuration


class Function(object):

    def __call__(self, *inputs):
        in_data = [x.data for x in inputs]
        outputs = self.forward(in_data)
        ret = [Variable(y) for y in outputs]

        if configuration.config.enable_backprop:
            self.rank = max([x.rank for x in inputs])
            for y in ret:
                y.set_creator(self)
            self.inputs = inputs
            self.outputs = ret

        return ret
        #return ret if len(ret) > 1 else ret[0]

    def forward(self, inputs):
        NotImplementedError()

    def backward(self, grad_outputs):
        NotImplementedError()


def grad(output):
    # clear all CG grad_var
    candidate_funcs = [output.creator]
    visited_funcs = set()
    while candidate_funcs:
        func = candidate_funcs.pop()
        if func in visited_funcs:
            continue
        visited_funcs.add(func)

        for x in func.inputs:
            x.cleargrad()  # clear grad

            creator = x.creator
            if creator is not None and creator not in visited_funcs:
                candidate_funcs.append(creator)

    if output.grad_var is None:
        output.grad = np.ones_like(output.data)
    output.backward()
