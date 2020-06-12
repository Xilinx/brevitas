import torch
from torch.autograd import Function

class QuantizedLinearPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, x, Wt, scale_factor, w_qnt_type, out_features, bias, in_scale, in_qnt_type):
        if in_scale is not None:
            if in_qnt_type is not None:
                x = g.op('Div', x, in_scale, activation_qnt_s = in_qnt_type)
            else:
                x = g.op('Div', x, in_scale)
        ret = g.op('MatMul', x, Wt, weight_qnt_s = w_qnt_type)
        if bias is not None:
            ret = g.op('Add', ret, bias)
        if scale_factor is not None:
            # TODO add info about scaling factor constraints as attributes here
            # (e.g. power of two, channel-wise or tensor-wise, ...)
            ret = g.op('Mul', ret, scale_factor)
        return ret

    @staticmethod
    def forward(ctx, x, Wt, scale_factor, w_qnt_type, out_features, bias, in_scale, in_qnt_type):
        return torch.empty(1, out_features, dtype = torch.float)

class QuantizedConv2dPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, x, W, scale_factor, qnt_type, out_shape, pads, strides, bias, kernel_shape, groups):
        ret = g.op('Conv', x, W, weight_qnt_s = qnt_type,
            kernel_shape_i = kernel_shape,
            pads_i = pads,
            strides_i = strides,
            group_i = groups,
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
    def forward(ctx, x, W, scale_factor, qnt_type, out_shape, pads, strides, bias, kernel_shape, groups):
        return torch.empty(out_shape, dtype = torch.float)

class QuantizedHardTanhPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, input, qnt_type, thres, bias, scale):
        if qnt_type == "BIPOLAR":
          return g.op('MultiThreshold', input, thres, domain_s = "finn",
                      out_dtype_s = qnt_type, out_scale_f = 2.0, out_bias_f = -1.0)
        else:
          ret = g.op('MultiThreshold', input, thres, domain_s = "finn",
                      out_dtype_s = qnt_type)
          if bias is not None:
              ret = g.op('Add', ret, bias)
          if scale is not None:
              ret = g.op('Mul', ret, scale)
          return ret

    @staticmethod
    def forward(ctx, input, qnt_type, thres, bias, scale):
        return input.clamp(0)

# Do we need a separate Place holder for this?
# Use QuantizedHardTanhPlaceholderFunction?
# keeping same interface
class QuantReLUPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, input, qnt_type, thres, bias, scale):
        ret = g.op('MultiThreshold', input, thres, domain_s = "finn",
                    out_dtype_s = qnt_type)
        if scale is not None:
            ret = g.op('Mul', ret, scale)
        return ret

    @staticmethod
    def forward(ctx, input, qnt_type, thres, bias, scale):
        return input.clamp(0)


class QuantAvgPool2dPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, input, out_shape, kernel, stride, signed, ibits, obits, scale, qnt_type):
        if scale is not None:
            input = g.op('Div', input, scale, activation_qnt_s = qnt_type)
        ret = g.op('QuantAvgPool2d', input, domain_s = "finn",
            kernel_i = kernel, stride_i = stride, signed_i = signed,
            ibits_i = ibits, obits_i = obits
        )
        if scale is not None:
            ret = g.op('Mul', ret, scale)
        return ret

    @staticmethod
    def forward(ctx, input, out_shape, kernel, stride, signed, ibits, obits, scale, qnt_type):
        return torch.empty(out_shape, dtype = torch.float)
