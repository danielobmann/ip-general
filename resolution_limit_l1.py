import odl
import numpy as np
from util import trig_transform
import matplotlib.pyplot as plt

# Gridsizes
Nx = 2048
xmax = np.pi

# Set up grid
space = odl.uniform_discr(min_pt=0, max_pt=xmax, shape=Nx, dtype='float64')
cosine = trig_transform(space, 'cosine')
ks = cosine.range.meshgrid[0]

beta = 2
sigma = 1. / np.abs(ks - 1) ** beta
m = odl.MultiplyOperator(cosine.range.element(sigma))

A = cosine.adjoint * m * cosine

func = lambda t: np.where(np.abs(t - 1) <= 0.1, 1., 0.) + np.where(np.abs(t - 2) <= 0.05, 1., 0.) + np.where(
    np.abs(t - 2.5) <= 0.01, 1., 0.)

f = space.element(func(space.meshgrid[0]))
g = A(f)

g.show()
coeff = cosine(g)
beta_est = 2
plt.semilogy(-ks, np.abs(coeff))
plt.semilogy(-ks, (1 - ks) ** (-beta_est))


def generate_noisy_data(data, eta=1e-2):
    noise = np.random.normal(0, eta, Nx)
    return data + noise


def l1_rec(data, alpha=1e-3, niter=5000, gamma=0.5):
    l2 = odl.solvers.L2NormSquared(A.range).translated(data) * A
    l1 = alpha * odl.solvers.L1Norm(space)
    x0 = space.zero()
    odl.solvers.nonsmooth.accelerated_proximal_gradient(x0, l1, l2, gamma=gamma, niter=niter)
    return x0


for eta in [1e-1, 1e-2, 0]:
    gnoisy = generate_noisy_data(g, eta=eta)
    coeff_noisy = cosine(gnoisy)
    delta = (g - gnoisy).norm()

    print("estimated resolution limit %f" % (np.sqrt(delta)))

    plt.figure()
    plt.loglog(-ks, np.abs(coeff))
    plt.loglog(-ks, np.abs(coeff_noisy))

    plt.title("delta = %f" % delta)

    for alpha in [1e-1, 1e-2, 1e-3, 1e-4]:
        temp = l1_rec(gnoisy, alpha=alpha)
        temp.show(title="alpha = %f" % alpha)
