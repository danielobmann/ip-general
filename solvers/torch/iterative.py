from torch import nn
from odl.contrib import torch as odl_torch


class ConjugateGradientNormalEquation(nn.Module):

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
