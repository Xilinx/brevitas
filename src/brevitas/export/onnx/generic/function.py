import torch
from torch.autograd import Function

from brevitas.function import binary_sign
from brevitas.core.bit_width import BitWidthConst
from brevitas.core.quant import IntQuant, TruncIntQuant
from brevitas.quant.solver.common import solve_float_to_int_impl_from_enum


DOMAIN_STRING = "onnx.brevitas"


class BrevitasBinaryQuantFn(Function):

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, narrow_range, signed, rounding_mode):
        ret = g.op(
            'BipolarQuant',
            x, scale,
            domain_s=DOMAIN_STRING)
        return ret

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, narrow_range, signed, rounding_mode):
        y = binary_sign(x) * scale
        return y



class BrevitasQuantFn(Function):

    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, narrow_range, signed, rounding_mode):
        ret = g.op(
            'Quant',
            x, scale, zero_point, bit_width,
            domain_s=DOMAIN_STRING,
            rounding_mode_s=rounding_mode,
            signed_i=int(signed),
            narrow_i=int(narrow_range))
        return ret

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, narrow_range, signed, rounding_mode):
        float_to_int_impl = solve_float_to_int_impl_from_enum(rounding_mode)
        quant = IntQuant(
            float_to_int_impl=float_to_int_impl(), narrow_range=narrow_range, signed=signed)
        y = quant(scale, zero_point, bit_width, x)
        return y


class BrevitasTruncFn(Function):

    @staticmethod
    def symbolic(g, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        ret = g.op(
            'Trunc',
            x, scale, zero_point, input_bit_width, output_bit_width,
            rounding_mode_s=rounding_mode,
            domain_s=DOMAIN_STRING)
        return ret

    @staticmethod
    def forward(ctx, x, scale, zero_point, input_bit_width, output_bit_width, rounding_mode):
        float_to_int_impl = solve_float_to_int_impl_from_enum(rounding_mode)
        trunc = TruncIntQuant(
            float_to_int_impl=float_to_int_impl(),
            bit_width_impl=BitWidthConst(int(output_bit_width)))
        y_tuple = trunc(x, scale, zero_point, input_bit_width)
        return y_tuple[0]