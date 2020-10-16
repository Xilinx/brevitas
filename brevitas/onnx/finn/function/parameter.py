import torch
from torch.autograd import Function


class QuantizedLinearPlaceholderFunction(Function):

    @staticmethod
    def symbolic(g, x, Wt, scale_factor, w_qnt_type, out_shape, bias, in_scale, in_qnt_type):
        if in_scale is not None:
            if in_qnt_type is not None:
                x = g.op('Div', x, in_scale, activation_qnt_s = in_qnt_type)
            else:
                x = g.op('Div', x, in_scale)
        ret = g.op('MatMul', x, Wt, weight_qnt_s = w_qnt_type)
        if bias is not None:
            ret = g.op('Add', ret, bias)
        if scale_factor is not None:
            # add extra info about scaling factors here
            ret = g.op('Mul', ret, scale_factor)
        return ret

    @staticmethod
    def forward(ctx, x, Wt, scale_factor, w_qnt_type, out_shape, bias, in_scale, in_qnt_type):
        return torch.empty(out_shape, dtype = torch.float)


class QuantizedConvNdPlaceholderFunction(Function):

    @staticmethod
    def symbolic(
            g, x, W, scale_factor, w_qnt_type, out_shape, pads, strides, bias, in_scale, in_qnt_type, kernel_shape, groups, dilations):
        if in_scale is not None:
            if in_qnt_type is not None:
                x = g.op('Div', x, in_scale, activation_qnt_s = in_qnt_type)
            else:
                x = g.op('Div', x, in_scale)
        ret = g.op(
            'Conv', x, W,
            weight_qnt_s=w_qnt_type,
            kernel_shape_i=kernel_shape,
            pads_i=pads,
            strides_i=strides,
            group_i=groups,
            dilations_i=dilations)
        if bias is not None:
            ret = g.op('Add', ret, bias)
        if scale_factor is not None:
            # add extra info about scaling factors here
            ret = g.op('Mul', ret, scale_factor)
        return ret

    @staticmethod
    def forward(
            ctx, x, W, scale_factor, qnt_type, out_shape, pads, strides, bias, in_scale, in_qnt_type, kernel_shape, groups, dilations):
        return torch.empty(out_shape, dtype = torch.float)