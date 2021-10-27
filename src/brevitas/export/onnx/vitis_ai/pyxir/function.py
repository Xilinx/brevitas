from abc import abstractmethod
from packaging import version

import torch
from torch.autograd import Function

from brevitas import torch_version


DOMAIN_STRING = 'pyxir.onnx'


class DPUQuantReLUFn(Function):

    @staticmethod
    def symbolic(
            g, x,
            output_shape,
            input_bit_width,
            input_scale,
            output_bit_width,
            output_scale):
        ret = g.op(
            f'{DOMAIN_STRING}::Relu', x,
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


class DPUQuantAvgPoolFn(Function):

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
                f'{DOMAIN_STRING}::GlobalAveragePool', x,
                vai_quant_s=['vai_quant_in', 'vai_quant_out'],
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale])
        else:
            ret = g.op(
                f'{DOMAIN_STRING}::AveragePool', x,
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
        return torch.empty(out_shape, dtype=torch.float, device=x.device)


class DPUQuantEltwiseAddFn(Function):

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
            f'{DOMAIN_STRING}::Add', x, y,
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


class DPUQuantMaxPoolFn(Function):

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
                f'{DOMAIN_STRING}::Pad', x,
                vai_quant_s=['vai_quant_in', 'vai_quant_out'],
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[input_bit_width, input_scale],
                pads_i=pads)
        ret = g.op(
            f'{DOMAIN_STRING}::MaxPool', x,
            kernel_shape_i=kernel_shape,
            strides_i=strides,
            auto_pad_s='VALID',
            dilations_i=dilations,
            ceil_mode_i=ceil_mode,
            vai_quant_s=['vai_quant_in', 'vai_quant_out'],
            vai_quant_in_i=[input_bit_width, input_scale],
            vai_quant_out_i=[output_bit_width, output_scale])
        return ret

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
        return torch.empty(out_shape, dtype=torch.float, device=x.device)


class DPUQuantConv2dFn(Function):

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
                f'{DOMAIN_STRING}::Pad', x,
                vai_quant_s=['vai_quant_in', 'vai_quant_out'],
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[input_bit_width, input_scale],
                pads_i=padding)
        vai_quant_s = ['vai_quant_in', 'vai_quant_out', 'vai_quant_weights']
        if int_bias is not None:
            vai_quant_s += ['vai_quant_biases']
            ret = g.op(
                f'{DOMAIN_STRING}::Conv', x,
                int_weight,
                int_bias,
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
                f'{DOMAIN_STRING}::Conv', x,
                int_weight,
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
        return torch.empty(out_shape, dtype=torch.float, device=x.device)


class DPUQuantLinearFn(Function):

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
                f'{DOMAIN_STRING}::Gemm', x,
                int_weight,
                int_bias,
                transB_i=1,
                vai_quant_s=vai_quant_s,
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale],
                vai_quant_weights_i=[weight_bit_width, weight_scale],
                vai_quant_biases_i=[bias_bit_width, bias_scale])
        elif int_bias is None and torch_version <= version.parse('1.4.0'):
            ret = g.op(
                f'{DOMAIN_STRING}::Gemm', x,
                int_weight,
                torch.tensor(0),  # workaround
                transB_i=1,
                vai_quant_s=vai_quant_s,
                vai_quant_in_i=[input_bit_width, input_scale],
                vai_quant_out_i=[output_bit_width, output_scale],
                vai_quant_weights_i=[weight_bit_width, weight_scale])
        else:
            ret = g.op(
                f'{DOMAIN_STRING}::Gemm', x,
                int_weight,
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
        return torch.empty(out_shape, dtype=torch.float, device=x.device)

