import odl
import numpy as np
from ops import RealPart, ImagPart


def conjugate_gradient(op, b, pre=None, niter=100, eps=1e-4, xtrue=None, inner=None):
    # see e.g. https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    # preconditioner pre is assumed to be the already inverted operator, i.e. pre(x) solves the system Cz = x
    if pre is None:
        pre = odl.IdentityOperator(op.domain)
    if inner is None:
        inner = op.domain.inner

    x0 = op.domain.zero()

    r = b.copy()
    d = pre(r)
    delta_new = inner(r, d)
    energy_norm = []
    if xtrue is not None:
        energy_norm.append(inner(op(x0 - xtrue), x0 - xtrue) ** 0.5)
    else:
        # Calculates shifted energy norm
        energy_norm.append(0.5 * inner(op(x0), x0) - inner(b, x0))

    residuum = []
    for k in range(niter):
        residuum.append(inner(r, r) ** 0.5)
        q = op(d)
        alpha = delta_new / inner(d, q)

        x0 = x0 + alpha * d
        r = r - alpha * q

        if xtrue is not None:
            # Add np.real(...) to avoid issues with rounding errors
            energy_norm.append(np.real(inner(op(x0 - xtrue), x0 - xtrue) ** 0.5))
            if energy_norm[-1] <= eps:
                return x0, energy_norm, residuum
        else:
            # if xtrue is not available algorithm should break if residuum is small enough
            energy_norm.append(np.real(0.5 * inner(op(x0), x0) - inner(b, x0)))
            if residuum[-1] <= eps:
                break

        s = pre(r)
        delta_old = delta_new
        delta_new = inner(r, s)
        d = s + delta_new / delta_old * d

    return x0, energy_norm, residuum


def trig_transform(domain, transformation_type='exp', axes=None):
    if axes is None:
        axes = tuple(range(len(domain.shape)))

    if transformation_type == 'exp':
        return odl.trafos.FourierTransform(domain=domain, axes=axes)
    else:
        domain_ext = odl.uniform_discr(min_pt=2 * domain.min_pt,
                                       max_pt=2 * domain.max_pt,
                                       shape=[2 * n for n in domain.shape],
                                       dtype=domain.dtype)
        resize = odl.discr.discr_ops.ResizingOperator(domain=domain,
                                                      range=domain_ext,
                                                      pad_mode='constant',
                                                      pad_const=0,
                                                      axes=axes)
        fourier = odl.trafos.FourierTransform(domain=domain_ext, axes=axes)

        if transformation_type == 'cosine':
            projection = np.sqrt(2) * RealPart(fourier.range)
        elif transformation_type == 'sine':
            projection = -np.sqrt(2) * ImagPart(fourier.range)
        else:
            projection = odl.IdentityOperator(fourier.range)

        return projection * fourier * resize


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


def get_radon_adjoint(radon, mu=1., nu=0.5, transformation_type='exp'):
    # This function returns the adjoint of the Radon transform mapping between H^mu and H^nu
    if mu == 0 and nu == 0:
        return radon.adjoint

    # In range of Radon transform we always have to use 'exp' to avoid any errors due to even/odd extensions
    sob1 = get_embedding_adjoint(space=radon.range, mu=0, nu=nu, axes=(1,), transformation_type='exp')
    sob2 = get_embedding_adjoint(space=radon.domain, mu=mu, nu=0, transformation_type=transformation_type)

    return sob2 * radon.adjoint * sob1


def dottest(op, adj, innerx, innery, N=20):
    # Test how large the difference between (op(x), y) and (x, adj(y)) is
    # This should give an idea of how good the given adjoint fits
    diff = []

    for i in range(N):
        x = np.random.normal(0, 1, op.domain.shape)
        xelem = op.domain.element(x)
        xelem /= innerx(xelem, xelem) ** 0.5
        y = np.random.normal(0, 1, adj.domain.shape)
        yelem = adj.domain.element(y)
        yelem /= innery(yelem, yelem) ** 0.5

        diff.append(innerx(xelem, adj(yelem)) - innery(op(xelem), yelem))
    return diff
