import odl
import numpy as np
from ops.odl.transform import trig_transform


def get_embedding_adjoint(space, mu=1., nu=0., axes=(0, 1), transformation_type='exp'):
    # The embedding is assumed to map from H^mu(space) to H^nu(space)
    # Its adjoint is then smoothing of degree mu - nu
    # The inverse of this adjoint operator is obtained by simply interchanging the exponents mu and nu
    # axes defines the axes along which the transform is taken
    transform = trig_transform(domain=space, axes=axes, transformation_type=transformation_type)
    freq_vars = np.meshgrid(*transform.range.meshgrid)
    weight = 1.
    for a in axes:
        weight += freq_vars[a] ** 2
    weight = weight ** (nu - mu)

    weight_elem = transform.range.element(weight.T)
    return transform.adjoint * odl.MultiplyOperator(weight_elem) * transform


def get_sobolev_inner(space, exp=1., axes=(0, 1), transformation_type='exp'):
    # If the exponent for the inner product is zero we just use the already implemented inner product
    # Otherwise we set up the transform and define the inner product by weighting one of the functions with the
    # appropriate weight in frequency-domain
    if exp == 0:
        return space.inner

    lam = get_embedding_adjoint(space=space, mu=0, nu=exp, axes=axes, transformation_type=transformation_type)

    def inner(f, g):
        return lam.domain.inner(f, lam(g))

    return inner
