import odl
from ops.odl.util import RealPart, ImagPart
import numpy as np


def trig_transform(domain, transformation_type='exp', axes=None):
    if axes is None:
        axes = tuple(range(len(domain.shape)))

    if transformation_type == 'exp':
        return odl.trafos.FourierTransform(domain=domain, axes=axes)
    else:
        domain_ext = odl.uniform_discr(min_pt=2 * domain.min_pt,
                                       max_pt=2 * domain.max_pt,
                                       shape=[2 * n for n in domain.shape],
                                       dtype=domain.dtype)
        resize = odl.discr.discr_ops.ResizingOperator(domain=domain,
                                                      range=domain_ext,
                                                      pad_mode='constant',
                                                      pad_const=0,
                                                      axes=axes)
        fourier = odl.trafos.FourierTransform(domain=domain_ext, axes=axes)

        if transformation_type == 'cosine':
            projection = np.sqrt(2) * RealPart(fourier.range)
        elif transformation_type == 'sine':
            projection = -np.sqrt(2) * ImagPart(fourier.range)
        else:
            projection = odl.IdentityOperator(fourier.range)

        return projection * fourier * resize
