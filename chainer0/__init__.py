from chainer0.variable import Variable
from chainer0.function import Function
from chainer0.function import grad

from chainer0.functions import basic_math

from chainer0.configuration import config  # NOQA

basic_math.install_variable_arithmetics()


config.enable_backprop = True
config.train = True