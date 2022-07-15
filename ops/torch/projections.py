from torch import nn
import torch

from odl.contrib import torch as odl_torch
from solvers.torch.iterative import ConjugateGradientNormalEquation


class KernelProjection(nn.Module):
    """
    Approximates the projection of x onto the kernel of operator
    """

    def __init__(self, operator, niter=100):
        super(KernelProjection, self).__init__()
        self.operator = odl_torch.OperatorModule(operator=operator)
        self.niter = niter
        self.cgne = ConjugateGradientNormalEquation(operator=operator, niter=niter)

    def forward(self, x):
        rhs = torch.zeros_like(self.operator(x))
        return self.cgne(x, rhs)


class RangeProjection(nn.Module):
    """
    Approximates the projection of y onto the closure of the range of operator
    """

    def __init__(self, operator, niter=100):
        super(RangeProjection, self).__init__()
        self.adjoint = odl_torch.OperatorModule(operator=operator.adjoint)
        self.niter = niter
        self.cgne = ConjugateGradientNormalEquation(operator=operator.adjoint, niter=niter)

    def forward(self, y):
        rhs = torch.zeros_like(self.adjoint(y))
        return y - self.cgne(y, rhs)
