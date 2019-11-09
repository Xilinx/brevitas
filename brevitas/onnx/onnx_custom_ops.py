import torch
from torch.autograd import Function

class QuantizedLinearPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, x, Wt, scale_factor, qnt_type, out_features):
        ret = g.op('MatMul', x, Wt, weight_qnt_s = qnt_type)
        if scale_factor is not None:
            # TODO add info about scaling factor constraints as attributes here
            # (e.g. power of two, channel-wise or tensor-wise, ...)
            ret = g.op('Mul', ret, scale_factor)
        return ret

    @staticmethod
    def forward(ctx, x, Wt, scale_factor, qnt_type, out_features):
        return torch.empty(1, out_features, dtype = torch.float)

class QuantizedConv2dPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, x, W, scale_factor, qnt_type, out_shape, pads, strides, bias, kernel_shape):
        ret = g.op('Conv', x, W, weight_qnt_s = qnt_type,
            kernel_shape_i = kernel_shape,
            pads_i = pads,
            strides_i = strides,
            group_i = 1,
            dilations_i = [1, 1]
        )
        if scale_factor is not None:
            # TODO add info about scaling factor constraints as attributes here
            # (e.g. power of two, channel-wise or tensor-wise, ...)
            ret = g.op('Mul', ret, scale_factor)
        if bias is not None:
            ret = g.op('Add', ret, bias)
        return ret

    @staticmethod
    def forward(ctx, x, W, scale_factor, qnt_type, out_shape, pads, strides, bias, kernel_shape):
        return torch.empty(out_shape, dtype = torch.float)

class QuantizedHardTanhPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, input, qnt_type, thres, bias, scale):
        if qnt_type == "BIPOLAR":
            # TODO ONNX Sign op returns 0 for a 0 input, which does not conform
            # to bipolar {-1, +1} quantization.
            ret = g.op('Sign', input, activation_qnt_s = qnt_type)
        else:
            ret = g.op('MultiThreshold', input, thres, domain_s = "finn")
            if bias is not None:
                ret = g.op('Add', ret, bias)
            if scale is not None:
                ret = g.op('Mul', ret, scale)
        return ret

    @staticmethod
    def forward(ctx, input, qnt_type, thres, bias, scale):
        return input.clamp(0)
