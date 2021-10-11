import torch
from torch.autograd import Function


class QLinearConvFn(Function):

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
            output_dtype,
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
            output_dtype,
            int_bias,
            out_shape,
            kernel_size,
            padding,
            stride,
            groups,
            dilation):
        return torch.empty(out_shape, dtype=output_dtype, device=int_x.device)


class QLinearMatMulFn(Function):

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
            output_dtype,
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
            output_dtype,
            out_shape):
        return torch.empty(out_shape, dtype=output_dtype, device=int_x.device)