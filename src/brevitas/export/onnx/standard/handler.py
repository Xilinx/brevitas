from abc import ABC


from brevitas.export.onnx.handler import ONNXBaseHandler
from brevitas.export.handler import BitWidthHandlerMixin, ZeroPointHandlerMixin


class StdONNXQuantLayerHandler(BitWidthHandlerMixin, ZeroPointHandlerMixin, ONNXBaseHandler, ABC):

    @classmethod
    def quant_axis(cls, scale):
        for i, s in enumerate(scale.shape):
            if s != 1:
                return i
        return None

