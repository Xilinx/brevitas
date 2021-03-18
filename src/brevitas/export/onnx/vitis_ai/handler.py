from abc import ABC
import math

from torch import Tensor

from brevitas.nn.quant_layer import QuantLayerMixin
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.export.onnx.handler import ONNXBaseHandler


class DPUQuantLayerHandler(ONNXBaseHandler, ABC):

    @staticmethod
    def neg_scalar_exponent_from_scale(scale: Tensor):
        if scale is None:
            raise RuntimeError("Scale cannot be None")
        if scale.view(-1).shape[0] != 1:
            raise RuntimeError("Only per-tensor scaling is currently supported")
        scale = scale.item()
        neg_exponent = - math.log2(scale)
        if not neg_exponent.is_integer():
            raise RuntimeError("Only power-of-two scale factors are supported")
        neg_exponent = int(neg_exponent)
        return neg_exponent

    @staticmethod
    def validate_8b_bit_width(bit_width: Tensor):
        if bit_width is None:
            raise RuntimeError("Bit width cannot be None")
        bit_width = int(bit_width.item())
        return bit_width

    @staticmethod
    def quant_input_scale(module: QuantLayerMixin):
        scale = module.quant_input_scale()
        return DPUQuantLayerHandler.neg_scalar_exponent_from_scale(scale)

    @staticmethod
    def quant_output_scale(module: QuantLayerMixin):
        scale = module.quant_output_scale()
        return DPUQuantLayerHandler.neg_scalar_exponent_from_scale(scale)

    @staticmethod
    def quant_input_bit_width(module: QuantLayerMixin):
        bit_width = module.quant_input_bit_width()
        return DPUQuantLayerHandler.validate_8b_bit_width(bit_width)


    @staticmethod
    def quant_output_bit_width(module: QuantLayerMixin):
        bit_width = module.quant_output_bit_width()
        return DPUQuantLayerHandler.validate_8b_bit_width(bit_width)

    @staticmethod
    def quant_output_shape(module: QuantLayerMixin):
        cached_out = module._cached_out  # TODO add shape property to the module
        if cached_out is None:
            raise RuntimeError("Caching of outputs is required")
        return cached_out.shape

    def prepare_from_cached_io(self, cached_io):
        cached_inp, cached_out = cached_io
        self.symbolic_kwargs = {
            'output_shape': cached_out.shape,
            'input_bit_width': self.validate_8b_bit_width(cached_inp.bit_width),
            'input_scale': self.neg_scalar_exponent_from_scale(cached_inp.scale),
            'output_bit_width': self.validate_8b_bit_width(cached_out.bit_width),
            'output_scale': self.neg_scalar_exponent_from_scale(cached_out.scale)
        }


class DPUQuantWeightBiasHandler(ABC):

    @staticmethod
    def int_weight(module: QuantWBIOL):
        return module.int_weight(float_datatype=False).detach()

    @staticmethod
    def quant_weight_bit_width(module: QuantWBIOL):
        bit_width = module.quant_weight_bit_width()
        return DPUQuantLayerHandler.validate_8b_bit_width(bit_width)

    @staticmethod
    def quant_weight_scale(module: QuantWBIOL):
        quant_weight_scale = module.quant_weight_scale()
        return DPUQuantLayerHandler.neg_scalar_exponent_from_scale(quant_weight_scale)

    @staticmethod
    def int_bias(module: QuantWBIOL):
        if module.bias is not None:
            return module.int_bias(float_datatype=False).detach()
        else:
            return None

    @staticmethod
    def quant_bias_bit_width(module: QuantWBIOL):
        if module.bias is not None:
            bit_width = module.quant_bias_bit_width()
            return DPUQuantLayerHandler.validate_8b_bit_width(bit_width)
        else:
            return None

    @staticmethod
    def quant_bias_scale(module: QuantWBIOL):
        if module.bias is not None:
            scale = module.quant_bias_scale()
            return DPUQuantLayerHandler.neg_scalar_exponent_from_scale(scale)
        else:
            return None



