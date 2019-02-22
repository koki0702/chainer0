import numpy as np
from chainer0.variable import Variable
from chainer0 import configuration


class Function(object):

    def __call__(self, *inputs):
        inputs = [x if isinstance(x, Variable)
                  else Variable(np.array(x))
                  for x in inputs]

        in_data = [x.data for x in inputs]
        out_data = self.forward(*in_data)

        if not isinstance(out_data, tuple):
            out_data = (out_data,)
        outputs = [Variable(y) for y in out_data]

        self.rank = max([x.rank for x in inputs])
        for y in outputs:
            y.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        NotImplementedError()

    def backward(self, grad_outputs):
        NotImplementedError()


def grad(outputs, inputs, enable_double_backprop=False):
    '''
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
    '''

    for input in inputs:
        input.grad_var = None

    for output in outputs:
        output.backward(enable_double_backprop)

    return [input.grad_var for input in inputs]