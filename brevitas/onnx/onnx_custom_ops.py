import torch
from torch.autograd import Function
from brevitas.onnx import quantization_annotation

# TODO find a better way to carry the quantization annotations --
# consider putting them in as attributes, then using an ONNX pass to
# move them into the graph.quantization_annotation

class QuantizedLinearPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, W, x, bw, out_features):
        quantization_annotation[W.uniqueName()] = str(bw)
        return g.op('MatMul', W, x, domain_s = "finn")

    @staticmethod
    def forward(ctx, W, x, bw, out_features):
        return torch.empty(1, out_features, dtype = torch.float)

class QuantizedHardTanhPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, input):
        ret = g.op('QuantizedHardTanh', input, domain_s = "finn")
        # TODO fix bitwidth
        quantization_annotation[ret.uniqueName()] = "1"
        return ret

    @staticmethod
    def forward(ctx, input):
        return input.clamp(0)
