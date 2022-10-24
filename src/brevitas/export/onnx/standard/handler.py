from abc import ABC

import torch

from brevitas.function.ops import min_int, max_int
from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.handler import BitWidthHandlerMixin, ZeroPointHandlerMixin

def to_0dim_if_scalar(tensor):
    if len(tensor.shape) == 1 and tensor.shape[0] == 1:
        tensor = tensor.view(()) # 0-Dim tensor
    return tensor

class StdONNXQuantLayerHandler(BitWidthHandlerMixin, ZeroPointHandlerMixin, ONNXBaseHandler, ABC):

    @classmethod
    def quant_axis(cls, scale):
        for i, s in enumerate(scale.shape):
            if s != 1:
                return i
        return None

    @classmethod
    def clip_symbolic_kwargs(cls, narrow, signed, bit_width):
        if narrow or bit_width < 8.:
            dtype = torch.int8 if signed else torch.uint8
            return {
                'int_min_val': min_int(signed, narrow, bit_width).to(dtype),
                'int_max_val': max_int(signed, narrow, bit_width).to(dtype)}
        else:
            return None
