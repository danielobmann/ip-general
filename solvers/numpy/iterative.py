import numpy as np


def _energy_norm(A, x):
    return np.dot(A @ x, x) ** 0.5


def cg(A, rhs, niter=10, x0=None, preconditioner=None, xtrue=None, eps=1e-10, return_info=False):
    # see e.g. https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
    # preconditioner pre is assumed to be the already inverted operator
    if preconditioner is None:
        preconditioner = np.eye(A.shape[0])

    if x0 is None:
        x0 = np.zeros(A.shape[0])

    r = rhs.copy()
    d = preconditioner @ r
    delta_new = np.dot(r, d)
    energy_norm = []
    if xtrue is not None:
        energy_norm.append(_energy_norm(A, x0 - xtrue))
    else:
        # Calculates shifted energy norm
        energy_norm.append(0.5 * np.dot(A @ x0, x0) - np.dot(rhs, x0))

    residuum = []
    for k in range(niter):
        residuum.append(np.dot(r, r) ** 0.5)
        q = A @ d
        alpha = delta_new / np.dot(d, q)

        x0 = x0 + alpha * d
        r = r - alpha * q

        if xtrue is not None:
            energy_norm.append(_energy_norm(A, x0 - xtrue))
            if energy_norm[-1] <= eps:
                break
        else:
            energy_norm.append(np.real(0.5 * np.dot(A @ x0, x0) - np.dot(rhs, x0)))
            if residuum[-1] <= eps:
                break

        s = preconditioner @ r
        delta_old = delta_new
        delta_new = np.dot(r, s)
        d = s + delta_new / delta_old * d
    if return_info:
        return x0, energy_norm, residuum
    return x0


def cgne(x0, A, rhs, niter=10, return_error=False):
    d0 = rhs - A @ x0
    p0 = A.T @ d0
    s0 = p0
    if return_error:
        err = [np.sqrt(np.sum(d0 ** 2))]

    for k in range(niter):
        q0 = A @ p0
        alpha = np.sum(s0 ** 2) / np.sum(q0 ** 2)
        x0 = x0 + alpha * p0
        d0 = d0 - alpha * q0
        if return_error:
            err.append(np.sqrt(np.sum(d0 ** 2)))
        s1 = A.T @ d0
        beta = np.sum(s1 ** 2) / np.sum(s0 ** 2)
        p0 = s1 + beta * p0
        s0 = s1
    if return_error:
        return x0, err
    return x0
