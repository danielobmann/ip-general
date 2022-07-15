import odl
import numpy as np


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
