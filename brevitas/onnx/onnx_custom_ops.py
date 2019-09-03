import torch
from torch.autograd import Function

class QuantizedLinearPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, W, x, bw, out_features):
        ret = g.op('MatMul', W, x, domain_s = "finn", weight_qnt_s = str(bw))
        return ret

    @staticmethod
    def forward(ctx, W, x, bw, out_features):
        return torch.empty(1, out_features, dtype = torch.float)

class QuantizedHardTanhPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, input):
        # TODO fix quantization info
        ret = g.op('QuantizedHardTanh', input, domain_s = "finn", activation_qnt_s = "1")
        return ret

    @staticmethod
    def forward(ctx, input):
        return input.clamp(0)
