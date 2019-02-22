import numpy as np
import heapq
import matplotlib.pyplot as plt
import chainer0
from chainer0 import Function, Variable
import chainer0.functions as F
from chainer0.computational_graph import get_dot_graph

x = Variable(np.array([1.0]), name='x')

#y = F.sin(x)
#y = (y + F.exp(x) - 0.5) * y
#y.backward()
y = F.tanh(x)
y.backward()


for i in range(3):
    gx = x.grad_var
    x.cleargrad()
    gx.backward()


txt = get_dot_graph(gx)
print(txt)