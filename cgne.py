from torch import nn
from odl.contrib import torch as odl_torch
import numpy as np


class ConjugateGradientNormalEquation(nn.Module):
    """
    CGNE is not a continuous operation for any number of iterations (in infinite dimensional spaces), since the noise
    can be shifted to the frequencies k+1, ... and in this case the error of CGNE(y) and CGNE(y_delta) is proportional
    to delta / sigma_m for any m.
    Thus, this layer actually has instabilities and should not be used for regularization purposes!

    Instead we can employ stopping criteria to ensure that CGNE is a regularization method. One such criterion is
    the discrepancy principle.
    """

    def __init__(self, operator, niter=10):
        super(ConjugateGradientNormalEquation, self).__init__()
        self.op = odl_torch.OperatorModule(operator=operator)
        self.adjoint = odl_torch.OperatorModule(operator=operator.adjoint)
        self.niter = niter
        self.correction_factor = operator.domain.weighting.const / operator.range.weighting.const

    def forward(self, x0, rhs):
        d0 = rhs - self.op(x0)
        p0 = self.adjoint(d0)
        s0 = p0

        for k in range(self.niter):
            q0 = self.op(p0)
            alpha = s0.norm() ** 2 / q0.norm() ** 2 * self.correction_factor
            x0 = x0 + alpha * p0
            d0 = d0 - alpha * q0
            s1 = self.adjoint(d0)
            beta = s1.norm() ** 2 / s0.norm() ** 2
            p0 = s1 + beta * p0
            s0 = s1
        return x0


def CGNE_numpy(x0, A, rhs, niter=10):
    d0 = rhs - A @ x0
    p0 = A.T @ d0
    s0 = p0

    for k in range(niter):
        q0 = A @ p0
        alpha = np.sum(s0 ** 2) / np.sum(q0 ** 2) ** 2
        x0 = x0 + alpha * p0
        d0 = d0 - alpha * q0
        s1 = A.T @ d0
        beta = np.sum(s1 ** 2) / np.sum(s0 ** 2)
        p0 = s1 + beta * p0
        s0 = s1
    return x0
