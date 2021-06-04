import torch
from torch.autograd import Function


class QuantAvgPool2dPlaceholderFunction(Function):

    @staticmethod
    def symbolic(g, x, out_shape, kernel, stride, signed, ibits, obits, scale, qnt_type):
        if scale is not None:
            x = g.op('Div', x, scale, activation_qnt_s=qnt_type)
        ret = g.op(
            'QuantAvgPool2d', x,
            domain_s="finn.custom_op.general",
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
