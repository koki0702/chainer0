import numpy as np

from chainer0.function import Function


class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, inputs):
        x = inputs[0]
        y = x.transpose(self.axes)
        return y,

    def backward(self, grad_vars):
        inv_axes = self.axes
        if inv_axes:
            axes_len = len(inv_axes)
            inv_axes = tuple(np.argsort([ax % axes_len for ax in inv_axes]))
        return Transpose(inv_axes)(grad_vars)


def transpose(x, axes=None):
    """Permute the dimensions of an input variable without copy.
        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`):
                Input variable to be transposed.
                A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
            axes (tuple of ints): By default, reverse the dimensions,
                otherwise permute the axes according to the values given.
        Returns:
            ~chainer.Variable: Variable whose axes are permuted.
        .. admonition:: Example
            >>> x = np.array([[[0, 1, 2], [3, 4, 5]]], np.float32)
            >>> x.shape
            (1, 2, 3)
            >>> y = F.transpose(x)  # reverse the dimensions
            >>> y.shape
            (3, 2, 1)
            >>> y.data
            array([[[0.],
                    [3.]],
            <BLANKLINE>
                   [[1.],
                    [4.]],
            <BLANKLINE>
                   [[2.],
                    [5.]]], dtype=float32)
            >>> y = F.transpose(x, axes=(1, 0, 2)) # swap 1st and 2nd axis
            >>> y.shape
            (2, 1, 3)
            >>> y.data
            array([[[0., 1., 2.]],
            <BLANKLINE>
                   [[3., 4., 5.]]], dtype=float32)
    """
    f = Transpose(axes)
    return f(x)[0]
