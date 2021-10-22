import torch
from torch.autograd import Function

from . import DOMAIN_STRING


class QuantizedLinearFn(Function):

    @staticmethod
    def symbolic(g, x, Wt, w_qnt_scale, b_qnt_scale, w_qnt_type, b_qnt_type, out_shape, bias):
        ret = g.op(f'{DOMAIN_STRING}::MatMul', x, Wt, weight_qnt_s=w_qnt_type)
        if w_qnt_scale is not None:
            ret = g.op('Mul', ret, w_qnt_scale)
        if bias is not None:
            if b_qnt_type is not None:
                assert b_qnt_scale is not None
                ret = g.op('Div', ret, b_qnt_scale)
                ret = g.op('{DOMAIN_STRING}::Add', ret, bias, bias_qnt_s=b_qnt_type)
                ret = g.op('Mul', ret, b_qnt_scale)
            else:
                ret = g.op('Add', ret, bias)
        return ret

    @staticmethod
    def forward(ctx, x, Wt, w_qnt_scale, b_qnt_scale, w_qnt_type, b_qnt_type, out_shape, bias):
        return torch.empty(out_shape, dtype=torch.float, device=x.device)


class QuantizedConvNdFn(Function):

    @staticmethod
    def symbolic(
            g, x, W, w_qnt_scale, b_qnt_scale, w_qnt_type, b_qnt_type, out_shape, pads, strides,
            bias, kernel_shape, groups, dilations):
        ret = g.op(
            f'{DOMAIN_STRING}::Conv', x, W,
            weight_qnt_s=w_qnt_type,
            kernel_shape_i=kernel_shape,
            pads_i=pads,
            strides_i=strides,
            group_i=groups,
            dilations_i=dilations)
        if w_qnt_scale is not None:
            ret = g.op('Mul', ret, w_qnt_scale)
        if bias is not None:
            if b_qnt_type is not None:
                assert b_qnt_scale is not None
                ret = g.op('Div', ret, b_qnt_scale)
                ret = g.op('{DOMAIN_STRING}::Add', ret, bias, bias_qnt_s=b_qnt_type)
                ret = g.op('Mul', ret, b_qnt_scale)
            else:
                ret = g.op('Add', ret, bias)
        return ret

    @staticmethod
    def forward(
            ctx, x, W, w_qnt_scale, b_qnt_scale, w_qnt_type, b_qnt_type, out_shape, pads, strides,
            bias, kernel_shape, groups, dilations):
        return torch.empty(out_shape, dtype=torch.float, device=x.device)