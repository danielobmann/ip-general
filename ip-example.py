"""
Illustration of instabilities in inverse problems.
"""

import numpy as np
import matplotlib.pyplot as plt

Ns = 512
xs = np.linspace(0, 1, Ns)
dx = xs[1] - xs[0]
A = np.tril(np.ones((Ns, Ns))) * dx

f = lambda x: np.sin(x)
fs = f(xs)

gs = A @ fs
gnoisy = gs + 1e-2 * np.random.normal(0, 1, Ns)

fsol = np.linalg.solve(A, gs)
fnoisy = np.linalg.solve(A, gnoisy)

plt.plot(xs, fs, label="true")
plt.plot(xs, fsol, label="noisefree inversion")
plt.plot(xs, fnoisy, label="noisy inverseion")
plt.legend()

"""
In this example A is the discretization of integration and as such the discretization of an ill-posed problem. This
ill-posedness is in the form of instabilities which can be observed in the plots where even small noise results in
an useless solution of the problem. This shows that some form of regularization is necessary even after discretization
of the problem.
"""
