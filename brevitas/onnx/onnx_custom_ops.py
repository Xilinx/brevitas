import torch
from torch.autograd import Function

class QuantizedLinearPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, W, x, qnt_type, out_features):
        ret = g.op('MatMul', W, x, domain_s = "finn", weight_qnt_s = qnt_type)
        return ret

    @staticmethod
    def forward(ctx, W, x, qnt_type, out_features):
        return torch.empty(1, out_features, dtype = torch.float)

class QuantizedHardTanhPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, input, qnt_type):
        ret = g.op('QuantizedHardTanh', input, domain_s = "finn", activation_qnt_s = qnt_type)
        return ret

    @staticmethod
    def forward(ctx, input, qnt_type):
        return input.clamp(0)
