import numpy as np
import matplotlib.pyplot as plt


def convolution_operator(ts, ys):
    nx = ts.__len__()
    ny = ys.__len__()
    func = lambda t, y: np.where(t <= y, 1., 0.) * t ** 2
    A = np.zeros((ny, nx))
    for i in range(ny):
        A[i, :] = func(ts, ys[i]) * (ts[1] - ts[0])
    return A


eps = 1e-2
tsim = np.linspace(-1 - eps, 1 + eps, 1234)
ysim = np.linspace(-1 - eps, 1 + eps, 1234)
Asim = convolution_operator(ts=tsim, ys=ysim)

trec = np.linspace(-1, 1, 256)
yrec = np.linspace(-1, 1, 256)
Arec = convolution_operator(ts=trec, ys=yrec)

ftrue = lambda t: np.where(t >= 0, 1., -1.)

xplus = ftrue(tsim)
data_high = Asim @ xplus
data_low = np.interp(yrec, ysim, data_high)
data_crime = Arec @ ftrue(trec)

xcrime = np.linalg.solve(Arec, data_crime)
xlow = np.linalg.solve(Arec, data_low)

plt.plot(trec, xcrime, label="crime")
plt.plot(trec, xlow, label="realistic")
plt.legend()

"""
Illustrates the problem of using the same operator for simulation of data and reconstruction of data.
If this is done then the reconstruction might be better than it should be. This is due to the negligence of 
modelling errors. Whenever modelling error is not disregarded then one can observe errors in the reconstruction.
Whenever the same operator is used for simulation and reconstruction it is referred to as the inverse crime. When
testing an algorithm it should always be avoided to get realistic simulation results and to avoid reporting too
optimistic results of the algorithm.
"""
