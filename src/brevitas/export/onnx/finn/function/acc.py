import torch
from torch.autograd import Function

from . import DOMAIN_STRING


class QuantAvgPool2dFn(Function):

    @staticmethod
    def symbolic(g, x, out_shape, kernel, stride, signed, ibits, obits, scale, qnt_type):
        if scale is not None:
            x = g.op('{DOMAIN_STRING}::Div', x, scale, activation_qnt_s=qnt_type)
        ret = g.op(
            f'{DOMAIN_STRING}::QuantAvgPool2d', x,
            kernel_i=kernel,
            stride_i=stride,
            signed_i=signed,
            ibits_i=ibits,
            obits_i=obits)
        if scale is not None:
            ret = g.op('Mul', ret, scale)
        return ret

    @staticmethod
    def forward(ctx, x, out_shape, kernel, stride, signed, ibits, obits, scale, qnt_type):
        return torch.empty(out_shape, dtype=torch.float, device=x.device)
