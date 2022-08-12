"""
Demonstration of the aliasing effect whenever the sampling rate is not chosen appropriately.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate as inter

func = lambda x, y: np.cos((x ** 2 + y ** 2) * 20 * np.pi)
samples = np.linspace(-1, 1, 2048)
xx, yy = np.meshgrid(samples, samples)
zz = func(xx, yy)

plt.figure()
plt.imshow(zz)

for ns in [16, 32, 64, 128]:
    # Sample given function with only ns points
    samplepoints = np.linspace(-1, 1, ns)
    xs, ys = np.meshgrid(samplepoints, samplepoints)
    temp = np.array([xs.flatten(), ys.flatten()])
    zs = func(*temp)

    # Reconstruct function from given samples using cubic splines
    finter = inter.griddata(temp.T, zs.flatten(), (xx, yy), method="cubic")
    plt.figure()
    plt.imshow(finter)
