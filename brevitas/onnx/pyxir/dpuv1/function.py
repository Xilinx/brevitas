import torch
from torch.autograd import Function


class QuantizedConv2dPlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x,
            int_weight,
            int_bias,
            out_shape,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale,
            weight_bit_width,
            weight_scale,
            bias_bit_width,
            bias_scale,
            kernel_size,
            padding,
            stride,
            groups,
            dilation):
        ret = g.op(
            'Conv', x,
            int_weight,
            int_bias,
            domain_s="pyxir_dpuv1",
            input_bit_width_i=input_bit_width,
            input_scale_i=input_scale,
            output_bit_width_i=output_bit_width,
            output_scale_i=output_scale,
            weight_bit_width_i=weight_bit_width,
            weight_scale_i=weight_scale,
            bias_bit_width_i=bias_bit_width,
            bias_scale_i=bias_scale,
            kernel_shape_i=kernel_size,
            pads_i=padding,
            strides_i=stride,
            group_i=groups,
            dilations_i=dilation)
        return ret

    @staticmethod
    def forward(
            ctx, x,
            int_weight,
            int_bias,
            out_shape,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale,
            weight_bit_width,
            weight_scale,
            int_bias_bit_width,
            int_bias_scale,
            kernel_size,
            padding,
            stride,
            groups,
            dilation):
        return torch.empty(out_shape, dtype=torch.float)


class QuantizedReLUPlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        ret = g.op(
            'ReLU', x,
            domain_s="pyxir_dpuv1",
            input_bit_width_i=input_bit_width,
            input_scale_i=input_scale,
            output_bit_width_i=output_bit_width,
            output_scale_i=output_scale)
        return ret

    @staticmethod
    def forward(
            ctx, x,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        return x.clamp(0.0)


class QuantizedPoolPlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x, int_weight, int_bias,
            out_shape,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        ret = g.op(
            'Pool', x, int_weight, int_bias,
            domain_s="pyxir_dpuv1",
            input_bit_width_i=input_bit_width,
            input_scale_i=input_scale,
            output_bit_width_i=output_bit_width,
            output_scale_i=output_scale)
        return ret

    @staticmethod
    def forward(
            ctx, x,
            out_shape,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        return torch.empty(out_shape, dtype=torch.float)


class QuantizedEltwisePlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x,
            out_shape,
            input0_bit_width,
            input0_scale,
            output_bit_width,
            output_scale):
        ret = g.op(
            'Eltwise', x,
            domain_s="pyxir_dpuv1",
            input0_bit_width_i=input0_bit_width,
            input0_scale_i=input0_scale,
            input1_bit_width_i=input0_bit_width,
            input1_scale_i=input0_scale,
            output_bit_width_i=output_bit_width,
            output_scale_i=output_scale)
        return ret

    @staticmethod
    def forward(
            ctx, x,
            out_shape,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        return torch.empty(out_shape, dtype=torch.float)