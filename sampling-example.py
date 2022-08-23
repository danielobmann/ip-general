import numpy as np
import matplotlib.pyplot as plt

Ns = 2 ** 8
bmax = 20
dx = np.pi / bmax
xs = dx * np.arange(Ns)


def sinc(x, b=1):
    return np.sinc(b * x / np.pi) * b


"""
According to the Nyquist sampling theorem we need a sampling rate with dx <= pi / b where b is the bandwidth of the 
signal to faithfully recover the signal. The function x -> sinc(b*x) has a bandwidth of b, so accordingly we
need a sampling rate of dx <= pi / b. 
With the given setup this is satisfied for b <= 20. As such we should be able to distinguish sincs of lower 
frequencies.
"""

plt.figure()
plt.plot(xs, sinc(xs, b=20))
plt.title("Samples of sinc")

plt.figure()
plt.plot(xs, sinc(xs, b=21))
plt.plot(xs, sinc(xs, b=-19) + 2*sinc(xs, b=20))
plt.title("Aliasing example")

"""
Note that b chosen such that dx = pi/b yields a delta peak. Moreover aliasing can be observed whenever b is chosen
too large, i.e. b > 20. In this case we get the "folding in" of the frequencies and a sinc with b = 21 results in
aliasing and it cannot be distinguished from a sum of lower frequency sincs.
"""
