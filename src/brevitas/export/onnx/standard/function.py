import torch
from torch.autograd import Function

from . import OPSET


AXIS_OPSET = 11


class DequantizeLinearFn(Function):

    @staticmethod
    def symbolic(
            g, x,
            input_scale,
            input_zero_point,
            input_axis):
        if input_axis is not None and OPSET >= AXIS_OPSET:
            ret = g.op(
                'DequantizeLinear', x,
                input_scale,
                input_zero_point,
                axis_i=input_axis)
        else:
            ret = g.op(
                'DequantizeLinear', x,
                input_scale,
                input_zero_point)
        return ret

    @staticmethod
    def forward(
            ctx, int_x,
            input_scale,
            input_zero_point,
            input_axis):
        return int_x.float()


class QuantizeLinearFn(Function):

    @staticmethod
    def symbolic(
            g, x,
            output_scale,
            ouput_zero_point,
            output_dtype,
            output_axis):
        if output_axis is not None and OPSET >= AXIS_OPSET:
            ret = g.op(
                'QuantizeLinear', x,
                output_scale,
                ouput_zero_point,
                axis_i=output_axis)
        else:
            ret = g.op(
                'QuantizeLinear', x,
                output_scale,
                ouput_zero_point)
        return ret

    @staticmethod
    def forward(
            ctx, x,
            output_scale,
            ouput_zero_point,
            output_dtype,
            output_axis):
        return x.type(output_dtype)

