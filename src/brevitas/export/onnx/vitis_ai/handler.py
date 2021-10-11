from abc import ABC
import math

from torch import Tensor

from brevitas.nn.quant_layer import QuantLayerMixin
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.export.handler import BitWidthHandlerMixin, ScaleHandlerMixin
from brevitas.export.onnx.handler import ONNXBaseHandler


class DPUQuantLayerHandler(ONNXBaseHandler, BitWidthHandlerMixin, ScaleHandlerMixin, ABC):

    @classmethod
    def quant_input_scale(cls, module: QuantLayerMixin):
        scale = module.quant_input_scale()
        return cls.validate_neg_scalar_int_exponent(scale)

    @classmethod
    def quant_output_scale(cls, module: QuantLayerMixin):
        scale = module.quant_output_scale()
        return cls.validate_neg_scalar_int_exponent(scale)

    @classmethod
    def quant_input_bit_width(cls, module: QuantLayerMixin):
        bit_width = module.quant_input_bit_width()
        return cls.validate_8b_bit_width(bit_width)


    @classmethod
    def quant_output_bit_width(cls, module: QuantLayerMixin):
        bit_width = module.quant_output_bit_width()
        return cls.validate_8b_bit_width(bit_width)

    @classmethod
    def quant_output_shape(cls, module: QuantLayerMixin):
        cached_out = module._cached_out
        if cached_out is None:
            raise RuntimeError("Caching of outputs is required")
        return cached_out.shape

    def prepare_from_cached_io(self, cached_io):
        cached_inp, cached_out = cached_io
        self.symbolic_kwargs = {
            'output_shape': cached_out.shape,
            'input_bit_width': self.validate_8b_bit_width(cached_inp.bit_width),
            'input_scale': self.validate_neg_scalar_int_exponent(cached_inp.scale),
            'output_bit_width': self.validate_8b_bit_width(cached_out.bit_width),
            'output_scale': self.validate_neg_scalar_int_exponent(cached_out.scale)
        }


class DPUQuantWBIOLHandler(DPUQuantLayerHandler):

    @classmethod
    def int_weight(cls, module: QuantWBIOL):
        return module.int_weight(float_datatype=False).detach()

    @classmethod
    def quant_weight_bit_width(cls, module: QuantWBIOL):
        bit_width = module.quant_weight_bit_width()
        return cls.validate_8b_bit_width(bit_width, le_then=True)

    @classmethod
    def quant_weight_scale(cls, module: QuantWBIOL):
        quant_weight_scale = module.quant_weight_scale()
        return cls.validate_neg_scalar_int_exponent(quant_weight_scale)

    @classmethod
    def int_bias(cls, module: QuantWBIOL):
        if module.bias is not None:
            return module.int_bias(float_datatype=False).detach()
        else:
            return None

    @classmethod
    def quant_bias_bit_width(cls, module: QuantWBIOL):
        if module.bias is not None:
            bit_width = module.quant_bias_bit_width()
            return DPUQuantLayerHandler.validate_8b_bit_width(bit_width, le_then=True)
        else:
            return None

    @classmethod
    def quant_bias_scale(cls, module: QuantWBIOL):
        if module.bias is not None:
            scale = module.quant_bias_scale()
            return cls.validate_neg_scalar_int_exponent(scale)
        else:
            return None



