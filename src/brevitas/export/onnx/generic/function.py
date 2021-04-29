import torch
from torch.autograd import Function


class QuantPlaceholderFunction(Function):

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, narrow_range, signed):
        ret = g.op(
            'Quant',
            x, scale, zero_point, bit_width,
            signed_i=int(signed),
            narrow_i=int(narrow_range))
        return ret

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, narrow_range, signed):
        return x


class TruncPlaceholderFunction(Function):

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width):
        ret = g.op(
            'Trunc',
            x, scale, zero_point, bit_width)
        return ret

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width):
        return x