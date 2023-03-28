"""
Example for quanta image sensor (QIS).

This is a sampling technique which makes use of the stochastic nature of the photon-counting process.
It uses only a single 0/1 switch to reflect whether a photon was counted or not. To make this work the technique
relies on temporal and/or spacial oversampling. For example, by improving the sampling factor by K we reduce the number
of expected photons from lam to lam / K. Thus, for K large enough we have that a simple 0/1-response is enough to
measure the expected value of lam which corresponds to the incident light.

Using a simple MLE one can estimate the actual value for each pixel separately.
"""

import numpy as np
import matplotlib.pyplot as plt

n = 512
K = 2 ** 12

# Set up true underlying light source
ts = np.linspace(0, np.pi, n)
c_true = 500 * np.sin(ts) ** 2 * np.exp(ts) * np.cos(ts ** 2) ** 2


def simulate_data(source, K=2 ** 12, quant=1):
    # Rescale to simulate K-fold oversampling in time
    sm = source / K
    B = np.zeros((K, n))

    # Simulate data using temporal oversampling with an over-sampling factor of K
    for k in range(K):
        ym = np.random.poisson(sm)
        B[k, :] = np.where(ym >= quant, 1, 0)
    return B


def estimate(B, S=K):
    K = B.shape[0]
    K1 = np.sum(B, axis=0)
    return np.where(K1 <= K * (1 - np.exp(-S / K)), -K * np.log(1 - K1 / K), S)


plt.figure()
plt.plot(c_true, label="c_true")


for K in [2**10, 2**11, 2**12, 2**13]:
    data = simulate_data(source=c_true, K=K)
    c_est = estimate(data, S=K)
    plt.plot(c_est, label=f"c_est, K={K}")

plt.legend()

