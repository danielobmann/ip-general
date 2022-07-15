# Examples of CGNE and its convergence for matrices with different behaviour of singular values

import numpy as np
from solvers.numpy.iterative import cgne
import matplotlib.pyplot as plt

n = 256
niter = 200

# Example of linear convergence
A = np.diag(1. / np.arange(1, n + 1))
x0 = np.ones(n)
xcg, err = cgne(x0, A, rhs=np.zeros_like(x0), niter=niter, return_error=True)
plt.semilogy(err)

# Example of linear convergence with non-trivial kernel
entries = [1 / i if i <= 200 else 0 for i in np.arange(1, n + 1)]
B = np.diag(entries)
x0 = np.ones(n)
xcg, err = cgne(x0, B, rhs=np.zeros_like(x0), niter=niter, return_error=True)
plt.semilogy(err)

# Example of slow convergence due to decay of singular values
q = 0.95
C = np.diag(q ** np.arange(0, n))

x0 = np.ones(n)
xcg, err = cgne(x0, C, rhs=np.zeros_like(x0), niter=niter, return_error=True)

x1 = np.ones(n)
x1[100:] = 0
xcg1, err1 = cgne(x1, C, rhs=np.zeros_like(x0), niter=niter, return_error=True)

plt.semilogy(err)
plt.semilogy(err1)
