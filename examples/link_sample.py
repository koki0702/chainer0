import numpy as np
import matplotlib.pyplot as plt
import chainer0.functions as F
import chainer0.links as L
from chainer0 import Function, Variable
from chainer0.functions import tanh



x = Variable(np.array(1))
y = Variable(np.array(2))
z = F.matmul(x,y)


x = Variable(np.array([1,2,3]))
W = np.arange(100).reshape(10,10)
y = F.embed_id(x, W)
print(x)
print(W)
print(y, y.shape)