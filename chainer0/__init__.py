from chainer0.variable import Variable
from chainer0.function import Function
from chainer0.function import grad
from chainer0.link import Link
from chainer0.link import Chain
from chainer0.functions import basic_math
from chainer0.functions import array
from chainer0.optimizer import Optimizer
from chainer0.configuration import config

from chainer0 import optimizers
from chainer0 import functions
from chainer0 import datasets


basic_math.install_variable_arithmetics()
array.install_variable_get_item()

config.enable_backprop = True
config.train = True