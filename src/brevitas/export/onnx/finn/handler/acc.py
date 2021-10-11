from torch import Tensor

from brevitas.nn import QuantAvgPool2d
from .base import FINNQuantIOHandler
from ..function.acc import QuantAvgPool2dFn


class FINNQuantAvgPool2dHandler(FINNQuantIOHandler):
    handled_layer = QuantAvgPool2d

    @staticmethod
    def quant_output_shape(module: QuantAvgPool2d):
        shape = FINNQuantIOHandler.quant_output_shape(module)
        if shape is None:
            raise RuntimeError("Caching of output shapes is required to export QuantAvgPool2d")
        return shape

    @staticmethod
    def quant_input_bit_width(module: QuantAvgPool2d):
        bit_width = FINNQuantIOHandler.quant_input_bit_width_tensor(module)
        if bit_width is None:
            raise RuntimeError("Caching of input bit width is required to export QuantAvgPool2d")
        return int(bit_width.item())

    @staticmethod
    def quant_output_bit_width(module: QuantAvgPool2d):
        bit_width = FINNQuantIOHandler.quant_output_bit_width_tensor(module)
        if bit_width is None:
            raise RuntimeError("Caching of output bit width is required to export QuantAvgPool2d")
        return int(bit_width.item())

    @staticmethod
    def quant_input_signed(module: QuantAvgPool2d) -> int:
        signed = FINNQuantIOHandler.quant_input_signed(module)
        if signed is None:
            raise RuntimeError("Output sign of QuantAvgPool2d is malformed")
        return int(signed)

    def prepare_for_export(self, module):
        self.symbolic_kwargs = {
            'out_shape': self.quant_output_shape(module),
            'kernel': module.kernel_size,
            'stride': module.stride,
            'signed': self.quant_input_signed(module),
            'ibits': self.quant_input_bit_width(module),
            'obits': self.quant_output_bit_width(module),
            'scale': self.quant_input_scale(module),
            'qnt_type': self.quant_input_type(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = QuantAvgPool2dFn.apply(inp, *self.symbolic_kwargs.values())
        return ret