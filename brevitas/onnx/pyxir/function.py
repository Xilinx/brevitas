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
        vai_quant_s = ['vai_quant_in', 'vai_quant_out', 'vai_quant_weights']
        if int_bias is not None:
            vai_quant_s += ['vai_quant_biases']
            ret = g.op(
                'Conv', x,
                int_weight,
                int_bias,
                domain_s="pyxir",
                vai_quant_s=vai_quant_s,
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale],
                vai_quant_weights_i=[weight_bit_width, weight_scale],
                vai_quant_biases_i=[bias_bit_width, bias_scale],
                kernel_shape_i=kernel_size,
                pads_i=padding,
                strides_i=stride,
                group_i=groups,
                dilations_i=dilation)
        else:
            ret = g.op(
                'Conv', x,
                int_weight,
                domain_s="pyxir",
                vai_quant_s=vai_quant_s,
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale],
                vai_quant_weights_i=[weight_bit_width, weight_scale],
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
            'Relu', x,
            domain_s="pyxir",
            vai_quant_s=['vai_quant_in', 'vai_quant_out'],
            vai_quant_in_i=[input_bit_width, input_scale],
            vai_quant_out_i=[output_bit_width, output_scale])
        return ret

    @staticmethod
    def forward(
            ctx, x,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        return x.clamp(0.0)


class QuantizedMaxPoolPlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x,
            out_shape,
            kernel_shape,
            pads,
            strides,
            ceil_mode,
            dilations,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        ret = g.op(
            'MaxPool', x,
            domain_s="pyxir",
            kernel_shape_i=kernel_shape,
            pads_i=pads,
            strides_i=strides,
            dilations_i=dilations,
            ceil_mode_i=ceil_mode,
            vai_quant_s=['vai_quant_in', 'vai_quant_out'],
            vai_quant_in_i=[input_bit_width, input_scale],
            vai_quant_out_i=[output_bit_width, output_scale])
        return ret

    @staticmethod
    def forward(
            ctx, x,
            out_shape,
            kernel_shape,
            pads,
            strides,
            ceil_mode,
            dilations,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        return torch.empty(out_shape, dtype=torch.float)


class QuantizedEltwiseAddPlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x, y,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        ret = g.op(
            'Add', x, y,
            domain_s="pyxir",
            vai_quant_s=['vai_quant_in', 'vai_quant_out'],
            vai_quant_in_i=[input_bit_width, input_scale],
            vai_quant_out_i=[output_bit_width, output_scale])
        return ret

    @staticmethod
    def forward(
            ctx, x, y,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        return x + y