import odl
import numpy as np
from util import trig_transform
import matplotlib.pyplot as plt

# Gridsizes
Nx = 4096
xmax = np.pi

# Set up grid
space = odl.uniform_discr(min_pt=0, max_pt=xmax, shape=Nx, dtype='float64')
cosine = trig_transform(space, 'cosine')
ks = cosine.range.meshgrid[0]

alpha = 2
sigma = 1. / np.abs(ks - 1) ** alpha
m = odl.MultiplyOperator(cosine.range.element(sigma))

A = cosine.adjoint * m * cosine

func = lambda t: np.where(np.abs(t - 1) <= 0.1, 1., 0.) + np.where(np.abs(t - 2) <= 0.05, 1., 0.) + np.where(
    np.abs(t - 2.5) <= 0.01, 1., 0.)

f = space.element(func(space.meshgrid[0]))
g = A(f)

g.show()
coeff = cosine(g)
beta = 2
plt.semilogy(-ks, np.abs(coeff))
plt.semilogy(-ks, (1 - ks) ** (-beta))


def generate_noisy_data(data, eta=1e-2):
    noise = np.random.normal(0, eta, Nx)
    return data + noise


def tsvd(data, k=10):
    coeff = cosine(data)
    coeff_inv = coeff / sigma
    coeff_inv[:(Nx + 1 - k)] = 0.
    return cosine.adjoint(coeff_inv)


"""
Since frequency has the inverse dimension of the space variable we get that the resolution limit is given as
c / k whenever we use only k values in the TSVD. To resolve a peak of width a we thus need c / k <= a.
For the cosine basis the c value can be estimated by the FWHM at around c = 2. Thus, in the example given above
we have with widths 0.2, 0.1 and 0.02 an estimation of k = 10, 20 and 100 to truthfully recover the width
of the peaks.

Moreover, noise hinders us from achieving the desired resolution. If the coefficients of the unperturbed data behave
like k**(-beta) and the noise level is delta, then the resolution is capped at around k**(-beta) <= delta, i.e.
k = delta**(-1 / beta).

In this example, the coefficients behave like k**(-beta) with beta = 2 and hence the resolution should be limited at
around delta**(-0.5).
"""

for eta in [1e-1, 1e-2, 0]:
    gnoisy = generate_noisy_data(g, eta=eta)
    coeff_noisy = cosine(gnoisy)
    delta = (g - gnoisy).norm()

    print("estimated resolution limit %f" % (np.sqrt(delta)))

    plt.figure()
    plt.loglog(-ks, np.abs(coeff))
    plt.loglog(-ks, np.abs(coeff_noisy))

    plt.title("delta = %f" % delta)

    for k in [10, 20, 50, 100]:
        temp = tsvd(gnoisy, k=k)
        temp.show(title="k = %f" % k)

