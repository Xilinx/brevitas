from packaging import version

import torch
from torch.autograd import Function

from brevitas import torch_version

DOMAIN_STRING = 'xir.onnx'


class XIRFixFn(Function):

    @staticmethod
    def symbolic(g, x, bit_width, fix_point, signed):
        ret = g.op(
            f'{DOMAIN_STRING}::Fix', x,
            bit_width_i=bit_width,
            fix_point_i=fix_point,
            signed_i=int(signed))
        return ret

    @staticmethod
    def forward(ctx, x, bit_width, fix_point, signed):
        return x


class XIRGemmFn(Function):

    @staticmethod
    def symbolic(g, x, weight, bias):
        if bias is not None:
            ret = g.op(f'{DOMAIN_STRING}::Gemm', x, weight, bias, transA_i=0, transB_i=1)
        elif bias is None and torch_version <= version.parse('1.4.0'):
            ret = g.op(f'{DOMAIN_STRING}::Gemm', x, weight, torch.tensor(0), transA_i=0, transB_i=1)
        else:
            ret = g.op(f'{DOMAIN_STRING}::Gemm', x, weight, transA_i=0, transB_i=1)
        return ret

    @staticmethod
    def forward(ctx, x, weight, bias):
        return torch.nn.functional.linear(x, weight, bias)


class XIRConv2dFn(Function):

    @staticmethod
    def symbolic(
            g, x, weight, bias, is_depthwise, kernel_size,
            padding, padding_type, stride, dilation, output_shape):
        if is_depthwise and bias is not None:
            ret = g.op(
                f'{DOMAIN_STRING}::DepthwiseConv2d', x, weight, bias,
                kernel_shape_i=kernel_size,
                padding_type_s=padding_type,
                pads_i=padding,
                strides_i=stride,
                dilations_i=dilation)
        elif is_depthwise and bias is None:
            ret = g.op(
                f'{DOMAIN_STRING}::DepthwiseConv2d', x, weight,
                kernel_shape_i=kernel_size,
                padding_type_s=padding_type,
                pads_i=padding,
                strides_i=stride,
                dilations_i=dilation)
        elif not is_depthwise and bias is not None:
            ret = g.op(
                f'{DOMAIN_STRING}::Conv2d', x, weight, bias,
                kernel_shape_i=kernel_size,
                padding_type_s=padding_type,
                pads_i=padding,
                strides_i=stride,
                dilations_i=dilation)
        else:
            ret = g.op(
                f'{DOMAIN_STRING}::Conv2d', x, weight,
                kernel_shape_i=kernel_size,
                padding_type_s=padding_type,
                pads_i=padding,
                strides_i=stride,
                dilations_i=dilation)
        return ret

    @staticmethod
    def forward(
            ctx, x, weight, bias, is_depthwise, kernel_size,
            padding, padding_type, stride, dilation, output_shape):
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)


class XIRConvTranpose2dFn(Function):

    @staticmethod
    def symbolic(
            g, x, weight, bias, is_depthwise, kernel_size, padding, padding_type, stride, dilation):
        if is_depthwise and bias is not None:
            ret = g.op(
                f'{DOMAIN_STRING}::DepthwiseConvTranpose2d', x, weight, bias,
                kernel_shape_i=kernel_size,
                padding_type_s=padding_type,
                pads_i=padding,
                strides_i=stride,
                dilations_i=dilation)
        elif is_depthwise and bias is None:
            ret = g.op(
                f'{DOMAIN_STRING}::DepthwiseConvTranpose2d', x, weight,
                kernel_shape_i=kernel_size,
                padding_type_s=padding_type,
                pads_i=padding,
                strides_i=stride,
                dilations_i=dilation)
        elif not is_depthwise and bias is not None:
            ret = g.op(
                f'{DOMAIN_STRING}::ConvTranspose2d', x, weight, bias,
                kernel_shape_i=kernel_size,
                padding_type_s=padding_type,
                pads_i=padding,
                strides_i=stride,
                dilations_i=dilation)
        else:
            ret = g.op(
                f'{DOMAIN_STRING}::ConvTranspose2d', x, weight,
                kernel_shape_i=kernel_size,
                padding_type_s=padding_type,
                pads_i=padding,
                strides_i=stride,
                dilations_i=dilation)
        return ret

    @staticmethod
    def forward(
            ctx, x, weight, bias, is_depthwise, kernel_size,
            padding, padding_type, stride, dilation, output_shape):
        return torch.empty(output_shape, dtype=x.dtype, device=x.device)