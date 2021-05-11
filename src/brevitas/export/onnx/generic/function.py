from torch.autograd import Function
from brevitas.core.quant import IntQuant, DecoupledIntQuant


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
        quant = IntQuant(narrow_range=narrow_range, signed=signed)
        x = quant(scale, zero_point, bit_width, x)
        return x


class DecoupledQuantPlaceholderFunction(Function):

    @staticmethod
    def symbolic(g, x, pre_scale, pre_zero_point, scale, zero_point, bit_width, narrow_range, signed):
        ret = g.op(
            'DecoupledQuant',
            x, pre_scale, pre_zero_point, scale, zero_point, bit_width,
            signed_i=int(signed),
            narrow_i=int(narrow_range))
        return ret

    @staticmethod
    def forward(ctx, x, pre_scale, pre_zero_point, scale, zero_point, bit_width, narrow_range, signed):
        quant = DecoupledIntQuant(narrow_range=narrow_range, signed=signed)
        x = quant(pre_scale, pre_zero_point, scale, zero_point, bit_width, x)
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