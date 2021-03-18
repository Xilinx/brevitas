from ..function import DPUQuantMaxPoolPlaceholderFunction
from ..function import DPUQuantConv2dPlaceholderFunction


class DPUv2QuantConv2dPlaceholderFunction(DPUQuantConv2dPlaceholderFunction):

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
        if ((isinstance(padding, int) and padding != 0)
                or (isinstance(padding, (list, tuple)) and any([p != 0 for p in padding]))):
            x = g.op(
                'Pad', x,
                domain_s="pyxir",
                vai_quant_s=['vai_quant_in', 'vai_quant_out'],
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[input_bit_width, input_scale],
                pads_i=padding)
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
                strides_i=stride,
                auto_pad_s='VALID',
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
                strides_i=stride,
                auto_pad_s='VALID',
                group_i=groups,
                dilations_i=dilation)
        return ret


class DPUv2QuantMaxPoolPlaceholderFunction(DPUQuantMaxPoolPlaceholderFunction):

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
        if ((isinstance(pads, int) and pads != 0)
                or (isinstance(pads, (list, tuple)) and any([p != 0 for p in pads]))):
            x = g.op(
                'Pad', x,
                domain_s="pyxir",
                vai_quant_s=['vai_quant_in', 'vai_quant_out'],
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[input_bit_width, input_scale],
                pads_i=pads)
        ret = g.op(
            'MaxPool', x,
            domain_s="pyxir",
            kernel_shape_i=kernel_shape,
            strides_i=strides,
            auto_pad_s='VALID',
            dilations_i=dilations,
            ceil_mode_i=ceil_mode,
            vai_quant_s=['vai_quant_in', 'vai_quant_out'],
            vai_quant_in_i=[input_bit_width, input_scale],
            vai_quant_out_i=[output_bit_width, output_scale])
        return ret