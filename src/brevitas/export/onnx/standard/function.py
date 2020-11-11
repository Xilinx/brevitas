import torch
from torch.autograd import Function
from . import OPSET

AXIS_OPSET = 11


class DequantizeLinearFunction(Function):

    @staticmethod
    def symbolic(
            g, x,
            input_scale,
            input_zero_point,
            axis):
        if axis is not None and OPSET >= AXIS_OPSET:
            ret = g.op(
                'DequantizeLinear', x,
                input_scale,
                input_zero_point,
                axis_i=axis)
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
            axis):
        return int_x.float()


class QuantizeLinearFunction(Function):

    @staticmethod
    def symbolic(
            g, x,
            output_scale,
            ouput_zero_point,
            axis):
        if axis is not None and OPSET >= AXIS_OPSET:
            ret = g.op(
                'QuantizeLinear', x,
                output_scale,
                ouput_zero_point,
                axis_i=axis)
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
            axis):
        return x.int()


class QLinearConvFunction(Function):

    @staticmethod
    def symbolic(
            g, int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            ouput_zero_point,
            int_bias,
            out_shape,
            kernel_size,
            padding,
            stride,
            groups,
            dilation):
        if int_bias is not None:
            ret = g.op(
                'QLinearConv', int_x,
                input_scale,
                input_zero_point,
                int_weight,
                weight_scale,
                weight_zero_point,
                output_scale,
                ouput_zero_point,
                int_bias,
                kernel_shape_i=kernel_size,
                pads_i=padding,
                strides_i=stride,
                group_i=groups,
                dilations_i=dilation)
        else:
            ret = g.op(
                'QLinearConv', int_x,
                input_scale,
                input_zero_point,
                int_weight,
                weight_scale,
                weight_zero_point,
                output_scale,
                ouput_zero_point,
                kernel_shape_i=kernel_size,
                pads_i=padding,
                strides_i=stride,
                group_i=groups,
                dilations_i=dilation)
        return ret

    @staticmethod
    def forward(
            ctx, int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            output_zero_point,
            bias,
            out_shape,
            kernel_size,
            padding,
            stride,
            groups,
            dilation):
        return torch.empty(out_shape).type(output_zero_point.dtype)


class QLinearMatMulFunction(Function):

    @staticmethod
    def symbolic(
            g, int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            ouput_zero_point,
            out_shape):
        ret = g.op(
            'QLinearMatMul', int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            ouput_zero_point)
        return ret

    @staticmethod
    def forward(
            ctx, int_x,
            input_scale,
            input_zero_point,
            int_weight,
            weight_scale,
            weight_zero_point,
            output_scale,
            output_zero_point,
            out_shape):
        return torch.empty(out_shape).type(output_zero_point.dtype)