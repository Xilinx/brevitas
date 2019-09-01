from torch.autograd import Function

# TODO find a better way to carry the quantization annotations --
# consider putting them in as attributes, then using an ONNX pass to
# move them into the graph.quantization_annotation

quantization_annotation = dict()

class QuantizedLinearPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, W, x, bw, out_features):
        #import pdb; pdb.set_trace()
        quantization_annotation[W.uniqueName()] = str(bw)
        return g.op('MatMul', W, x)

    @staticmethod
    def forward(ctx, W, x, bw, out_features):
        return torch.empty(1, out_features, dtype = torch.float)

class QuantizedHardTanhPlaceholderFunction(Function):
    @staticmethod
    def symbolic(g, input):
        ret = g.op('QuantizedHardTanh', input, my_special_attr_s = "hello")
        # insert quantization annotation for the resulting tensor, TODO fix bitwidth
        quantization_annotation[ret.uniqueName()] = "1"
        return ret

    @staticmethod
    def forward(ctx, input):
        return input.clamp(0)


# TODO add flag-controlled alternative forward path to relevant modules instead
# class QuantizedHardTanhPlaceholder(nn.Module):
#     def __init__(self):
#         super(QuantizedHardTanhPlaceholder, self).__init__()
#
#     def forward(self, x):
#         return QuantizedHardTanhPlaceholderFunction.apply(x)
#
# class QuantizedLinearPlaceholder(nn.Module):
#     def __init__(self, quantized_linear):
#         super(QuantizedLinearPlaceholder, self).__init__()
#         self.in_features = quantized_linear.in_features
#         self.out_features = quantized_linear.out_features
#         # compute the quantized weights
#         W, s, bitwidth = quantized_linear.weight_quant(quantized_linear.weight)
#         W = W.detach().numpy().reshape(self.out_features, self.in_features)
#         s = s.detach().numpy()
#         s = s.reshape(s.size, 1)
#         W = W / s
#         self.W = torch.from_numpy(W)
#         self.bitwidth = bitwidth.item()
#
#    def forward(self, x):
#        return QuantizedLinearPlaceholderFunction.apply(self.W, x, self.bitwidth, self.out_features)
