import math
import numpy as np

import chainer0
import chainer0.functions as F
from chainer0 import Variable

try:
    import scipy.linalg
    available_cpu = True
except ImportError as e:
    available_cpu = False
    _import_error = e

ENTROPYC = 0.5 * math.log(2 * math.pi * math.e)
LOGPROBC = - 0.5 * math.log(2 * math.pi)
PROBC = 1. / (2 * math.pi) ** 0.5



class TriangularInv(chainer0.Function):

    def __init__(self, lower):
        self.lower = lower

    def forward(self, x):
        if not available_cpu:
            raise ImportError('SciPy is not available. Forward computation'
                              ' of triangular_inv in CPU can not be done.' +
                              str(_import_error))

        invx = scipy.linalg.solve_triangular(
            x, np.eye(len(x), dtype=x.dtype), lower=self._lower)
        return invx

class MultivariateNormal:

    """MultivariateNormal Distribution.
    The probability density function of the distribution is expressed as
    .. math::
        p(x;\\mu,V) = \\frac{1}{\\sqrt{\\det(2\\pi V)}}
            \\exp\\left(-\\frac{1}{2}(x-\\mu) V^{-1}(x-\\mu)\\right)
    Args:
        loc (:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the location :math:`\\mu`.
        scale_tril (:class:`~chainer.Variable` or :ref:`ndarray`): Parameter of
            distribution representing the scale :math:`L` such that
            :math:`V=LL^T`.
    """


    def __init__(self, loc, scale_tril):
        loc = Variable(loc) if not isinstance(loc, Variable) else loc
        scale_tril = Variable(scale_tril) if not \
            isinstance(scale_tril, Variable) else scale_tril

        self.loc = loc
        self.scale = scale_tril


    @property
    def entropy(self):
        return self.log_scale + ENTROPYC

    def prob(self, x):
        return (PROBC / self.scale) * F.exp(
            - 0.5 * (x - self.loc) ** 2 / self.scale ** 2)

    def log_prob(self, x):
        2 * np.pi *

    def log_prob2(self, x):
        return LOGPROBC - self.log_scale \
            - 0.5 * (x - self.loc) ** 2 / self.scale ** 2

    def sample_n(self, n):
        eps = np.random.standard_normal((n,) + self.loc.shape)
        return self.loc + self.scale * eps
