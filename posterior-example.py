"""
Example of a simple Bayesian inversion process.
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
cmap = "inferno"
eps = 1e-1  # defines the condition of the forward operator, i.e. cond(A) = 1/eps
sigma = 1  # defines the standard-deviation of the noise
rho = 3  # defines the a-priori standard-deviation

# Define simple forward operator
A = np.array([[1, 0], [0, eps]])

# Simulate data using white noise model
xtrue = np.array([1, 1])
ytrue = A @ xtrue
yobs = ytrue + sigma * np.random.normal(0, 1, A.shape[0])

# Plotting parameters
ns = 1024
xmin = -5
xmax = 5
xs = np.linspace(xmin, xmax, ns)
xvec = np.stack(np.meshgrid(xs, xs)).reshape((A.shape[1], -1))

# Calculate prior and likelihood
prior = np.exp(- np.sum(xvec ** 2, axis=0) / (2 * rho ** 2)).reshape((ns, ns))
res = (A @ xvec - yobs.reshape((A.shape[0], 1)))
likelihood = np.exp(- np.sum(res ** 2, axis=0) / (2 * sigma ** 2)).reshape((ns, ns))

"""
Plots show the prior which is centered on 0 and is spread out depending on the choice of rho, the likelihood which
is an ellipse where length along each axis depends on the singular-values of A and hence can be chosen by selecting
eps appropriately. Finally, the posterior is just the product. 

One can see that the prior helps the likelihood by restricting the size and hence essentially making the ellipse
tighter. Depending on the uncertainty parameters (rho and sigma) one can get different sizes of ellipses. Larger 
uncertainty yields larger ellipses reflecting the uncertainty in the posterior.
"""

plt.figure()
plt.imshow(np.flipud(prior), extent=[xmin, xmax, xmin, xmax], cmap=cmap)
plt.title("Prior")

plt.figure()
plt.imshow(np.flipud(likelihood), extent=[xmin, xmax, xmin, xmax], cmap=cmap)
plt.title("Likelihood")

plt.figure()
plt.imshow(np.flipud(likelihood * prior), extent=[xmin, xmax, xmin, xmax], cmap=cmap)
plt.title("Posterior")

"""
Sampling of posterior using MCMC method. This is a really simple method to sample a given posterior distribution.
With enough samples the histogram of the samples starts to look like to correct posterior as defined above.
The advantage of this method is, that one can use these samples to calculate many different things, e.g. conditional
expectation or use it to minimize some Bayesian cost function.
"""


def mcmc(posterior, gamma=1, n=1000, dim=1, burn_in_factor=0.3):
    x = np.zeros(dim)
    samples = []
    burn_in = int(n * burn_in_factor)
    for k in range(n):
        pi_x = posterior(x)
        y = x + np.random.normal(0, gamma, dim)
        pi_y = posterior(y)
        acceptance = np.minimum(1, pi_y / pi_x)
        if k > burn_in:
            if np.random.uniform(0, 1) <= acceptance:
                x = y
                samples.append(y)
    return np.stack(samples)


def posterior(x, y):
    prior = np.exp(- np.sum(x ** 2) / (2 * rho ** 2))
    res = (A @ x - y)
    likelihood = np.exp(- np.sum(res ** 2) / (2 * sigma ** 2))
    return likelihood * prior


post = lambda x: posterior(x, y=yobs)
samples = mcmc(post, dim=2, n=10 ** 6)  # Sampling takes some time
plt.figure()
plt.hist2d(samples[:, 0], samples[:, 1], range=[[xmin, xmax], [xmin, xmax]], cmap=cmap, bins=50)
plt.title("Empirical posterior")
