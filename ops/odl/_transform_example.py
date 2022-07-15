from ops.odl.transform import trig_transform
import odl
import numpy as np

N = 1024
space = odl.uniform_discr(0, np.pi, N)
sine = trig_transform(domain=space, transformation_type='sine')

t = space.meshgrid[0]
f = lambda s: np.exp(-s ** 2)
fs = f(t)
coeff = sine(fs)

# Use to solve lap(u) = f with Dirichlet boundary conditions
k = sine.range.meshgrid[0]
ucoeff = coeff / k ** 2
ucoeff[-1] = 0.
u = sine.adjoint(ucoeff)
u.show()
