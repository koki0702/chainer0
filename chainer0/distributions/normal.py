import math
import numpy as np
import chainer0.functions as F
from chainer0 import Variable


ENTROPYC = 0.5 * math.log(2 * math.pi * math.e)
LOGPROBC = - 0.5 * math.log(2 * math.pi)
PROBC = 1. / (2 * math.pi) ** 0.5


class Normal:

    """Normal Distribution.

    The probability density function of the distribution is expressed as

    .. math::
        p(x;\\mu,\\sigma) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}}
            \\exp\\left(-\\frac{(x-\\mu)^2}{2\\sigma^2}\\right)
    Args:
        loc(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the location :math:`\\mu`. This is the
            mean parameter.
        scale(:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the scale :math:`\\sigma`.

    """

    def __init__(self, loc, scale):
        loc = Variable(loc) if not isinstance(loc, Variable) else loc
        scale = Variable(scale) if not isinstance(scale, Variable) else scale

        self.loc = loc
        self.scale = scale
        self.log_scale = F.log(scale)

    @property
    def entropy(self):
        return self.log_scale + ENTROPYC

    def prob(self, x):
        return (PROBC / self.scale) * F.exp(
            - 0.5 * (x - self.loc) ** 2 / self.scale ** 2)

    def log_prob(self, x):
        return LOGPROBC - self.log_scale \
            - 0.5 * (x - self.loc) ** 2 / self.scale ** 2

    def sample_n(self, n):
        eps = np.random.standard_normal((n,) + self.loc.shape)
        return self.loc + self.scale * eps
