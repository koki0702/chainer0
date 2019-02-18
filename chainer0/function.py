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

        self.rank = max([x.rank for x in inputs])
        self.inputs = inputs

        if isinstance(out_data, tuple) or isinstance(out_data, list):
            outputs = [Variable(y) for y in out_data]
            for y in outputs:
                y.set_creator(self)
            self.outputs = outputs
            return outputs
        else:
            output = Variable(out_data)
            output.set_creator(self)
            self.output = output
            self.outputs = [output]
            return output

        #return outputs if len(outputs) > 1 else outputs[0]

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
