from abc import abstractmethod
from packaging import version

import torch
from torch.autograd import Function

from brevitas import torch_version


class DPUQuantReLUPlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x,
            output_shape,
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
            output_shape,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        return x


class DPUQuantAvgPoolPlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x,
            kernel_shape,
            strides,
            pads,
            out_shape,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        if kernel_shape == [1, 1]:  # passthrough in case it does nothing
            return x
        if list(out_shape[2:]) == [1, 1]:
            ret = g.op(
                'GlobalAveragePool', x,
                domain_s="pyxir",
                vai_quant_s=['vai_quant_in', 'vai_quant_out'],
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale])
        else:
            ret = g.op(
                'AveragePool', x,
                domain_s="pyxir",
                kernel_shape_i=kernel_shape,
                strides_i=strides,
                pads_i=pads,
                vai_quant_s=['vai_quant_in', 'vai_quant_out'],
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale])
        return ret

    @staticmethod
    def forward(
            ctx, x,
            kernel_shape,
            strides,
            pads,
            out_shape,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        return torch.empty(out_shape, dtype=torch.float)


class DPUQuantEltwiseAddPlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x, y,
            input_bit_width,
            input_scale,
            other_bit_width,
            other_scale,
            output_bit_width,
            output_scale):
        ret = g.op(
            'Add', x, y,
            domain_s="pyxir",
            vai_quant_s=['vai_quant_in', 'vai_quant_out'],
            vai_quant_in_i=[input_bit_width, input_scale, other_bit_width, other_scale],
            vai_quant_out_i=[output_bit_width, output_scale])
        return ret

    @staticmethod
    def forward(
            ctx, x, y,
            input_bit_width,
            input_scale,
            other_bit_width,
            other_scale,
            output_bit_width,
            output_scale):
        return x


class DPUQuantMaxPoolPlaceholderFunction(Function):

    @staticmethod
    @abstractmethod
    def symbolic(
            ctx, x,
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
        pass

    @staticmethod
    def forward(
            ctx, x,
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
        return torch.empty(out_shape, dtype=torch.float)


class DPUQuantConv2dPlaceholderFunction(Function):

    @staticmethod
    @abstractmethod
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
        pass

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


class DPUQuantLinearPlaceholderFunction(Function):

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
            bias_scale):
        vai_quant_s = ['vai_quant_in', 'vai_quant_out', 'vai_quant_weights']
        if int_bias is not None:
            vai_quant_s += ['vai_quant_biases']
            ret = g.op(
                'Gemm', x,
                int_weight,
                int_bias,
                domain_s="pyxir",
                transB_i=1,
                vai_quant_s=vai_quant_s,
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale],
                vai_quant_weights_i=[weight_bit_width, weight_scale],
                vai_quant_biases_i=[bias_bit_width, bias_scale])
        elif int_bias is None and torch_version <= version.parse('1.4.0'):
            ret = g.op(
                'Gemm', x,
                int_weight,
                torch.tensor(0),  # workaround
                domain_s="pyxir",
                transB_i=1,
                vai_quant_s=vai_quant_s,
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale],
                vai_quant_weights_i=[weight_bit_width, weight_scale])
        else:
            ret = g.op(
                'Gemm', x,
                int_weight,
                domain_s="pyxir",
                transB_i=1,
                vai_quant_s=vai_quant_s,
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale],
                vai_quant_weights_i=[weight_bit_width, weight_scale])
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
            int_bias_scale):
        return torch.empty(out_shape, dtype=torch.float)