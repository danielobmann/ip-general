"""
Data separation example using compressed sensing.
The goal is to separate some function f into two parts f_1 and f_2, where it is assumed that f_i can be sparsely
represented using A_i. Hence we get f = A_1 c_1 + A_2 c_2 where c_i is sparse.

For this demonstration we use a function which decomposes into sines and peaks. The sine part is then sparse in the
sine basis and the peaks are sparse in the standard basis.
"""

import numpy as np
from ops.odl.transform import trig_transform
import odl

N = 1024
space = odl.uniform_discr(0, np.pi, N)
sine = trig_transform(domain=space, transformation_type='sine')


def peak(a, delta):
    return lambda t: np.where(a <= t, 1., 0.) * np.where(t <= a + delta, 1., 0.)


def func(t):
    peak1 = peak(1, 0.05)
    peak2 = peak(2, 0.05)
    return 3 * np.sin(4 * t) + 5 * np.sin(20 * t) + 10 * peak1(t) + 10 * peak2(t)


data = space.element(func(space.meshgrid[0]))
data.show()

# Set up operator, regularizer and data-discrepancy term
forward_op = odl.operator.pspace_ops.ReductionOperator(odl.IdentityOperator(space), sine.adjoint)
domain = forward_op.domain

reg_peak = 0.1 * odl.solvers.L1Norm(space)
reg_sine = 0.01 * odl.solvers.L1Norm(sine.adjoint.domain)

regularizer = odl.solvers.SeparableSum(reg_peak, reg_sine)
l2_norm_sq = odl.solvers.L2NormSquared(space=space).translated(data)
data_discrepancy = l2_norm_sq * forward_op


c_rec = forward_op.domain.zero()
odl.solvers.accelerated_proximal_gradient(c_rec, f=regularizer, g=data_discrepancy, niter=1000, gamma=0.2)

# Visualize the two parts into which the function has been decomposed
f_peaks = c_rec.parts[0]
f_sine = sine.adjoint(c_rec.parts[1])

f_peaks.show()
f_sine.show()
