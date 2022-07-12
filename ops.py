import odl
from odl.operator.operator import Operator

"""This is an implementation of the real and imaginary part which fixes the problem with the domains of the adjoint.
Before the problem was that one could not do real.adjoint * real or imag.adjoint * imag, where real and imag denote
the corresponding operators. With the implementation as below this is possible and real and imag behave as expected.
"""


class RealPart(Operator):
    """Operator that extracts the real part of a vector.

    Implements::

        RealPart(x) == x.real
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `TensorSpace`
            Space in which the real part should be taken, needs to implement
            ``space.real_space``.

        """
        real_space = space.real_space
        super(RealPart, self).__init__(space, real_space, linear=True)

    def _call(self, x):
        """Return ``self(x)``."""
        return x.real

    def derivative(self, x):
        """Return the derivative operator.
        """
        return self

    @property
    def inverse(self):
        """Return the (pseudo-)inverse.
        """

        return odl.ComplexEmbedding(self.range, scalar=1)

    @property
    def adjoint(self):
        """Return the (left) adjoint.
        """
        return odl.ComplexEmbedding(self.range, scalar=1)


class ImagPart(Operator):
    """Operator that extracts the imaginary part of a vector.

    Implements::

        ImagPart(x) == x.imag
    """

    def __init__(self, space):
        """Initialize a new instance.

        Parameters
        ----------
        space : `TensorSpace`
            Space in which the imaginary part should be taken, needs to
            implement ``space.real_space``.

        """
        self.real_space = space.real_space
        super(ImagPart, self).__init__(space, self.real_space, linear=True)

    def _call(self, x):
        return x.imag

    def derivative(self, x):
        """Return the derivative operator in the.
        """
        return self

    @property
    def inverse(self):
        """Return the pseudoinverse.
        """
        return odl.ComplexEmbedding(self.range, scalar=1j)

    @property
    def adjoint(self):
        """Return the (left) adjoint.
        """
        return odl.ComplexEmbedding(self.range, scalar=1j)
