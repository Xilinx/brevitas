import torch
from torch.autograd import Function

class QuantizedLinearPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, W, x, scale_factor, qnt_type, out_features):
        ret = g.op('MatMul', W, x, domain_s = "finn", weight_qnt_s = qnt_type)
        if scale_factor is not None:
            # TODO add info about scaling factor constraints as attributes here
            # (e.g. power of two, channel-wise or tensor-wise, ...)
            ret = g.op('Mul', ret, scale_factor)
        return ret

    @staticmethod
    def forward(ctx, W, x, scale_factor, qnt_type, out_features):
        return torch.empty(1, out_features, dtype = torch.float)

class QuantizedHardTanhPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, input, qnt_type):
        if qnt_type == "BIPOLAR":
            # TODO ONNX Sign op returns 0 for a 0 input, which does not conform
            # to bipolar {-1, +1} quantization.
            ret = g.op('Sign', input, domain_s = "finn", activation_qnt_s = qnt_type)
        else:
            ret = g.op('QuantizedHardTanh', input, domain_s = "finn", activation_qnt_s = qnt_type)
        return ret

    @staticmethod
    def forward(ctx, input, qnt_type):
        return input.clamp(0)
