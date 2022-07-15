import numpy as np
from solvers.numpy.iterative import cgne


def kernel_projection(x, A, niter=100):
    rhs = np.zeros(A.shape[1])
    return cgne(x0=x, A=A, rhs=rhs, niter=niter)


def range_projection(y, A, niter=100):
    rhs = np.zeros(A.shape[0])
    return y - cgne(x0=y, A=A.T, rhs=rhs, niter=niter)
