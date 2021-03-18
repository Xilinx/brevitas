from ..function import DPUQuantMaxPoolPlaceholderFunction
from ..function import DPUQuantConv2dPlaceholderFunction


class DPUv1QuantConv2dPlaceholderFunction(DPUQuantConv2dPlaceholderFunction):

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


class DPUv1QuantMaxPoolPlaceholderFunction(DPUQuantMaxPoolPlaceholderFunction):

    @staticmethod
    def symbolic(
            g, x,
            kernel_shape,
            pads,
            strides,
            ceil_mode,
            dilations,
            out_shape,
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


