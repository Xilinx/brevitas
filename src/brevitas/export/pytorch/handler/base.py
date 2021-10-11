from abc import ABC, abstractmethod

import torch
from torch import Tensor
from brevitas.export.handler import BaseHandler, BitWidthHandlerMixin, ZeroPointHandlerMixin

SCALAR_SHAPE = ()


def _is_scalar(x: Tensor):
    return x.shape == SCALAR_SHAPE


class PytorchQuantLayerHandler(BaseHandler, BitWidthHandlerMixin, ZeroPointHandlerMixin, ABC):

    @classmethod
    @abstractmethod
    def explicit_output_dtype(cls) -> bool:
        pass

    @classmethod
    @abstractmethod
    def prepare_qf(cls, module):
        pass

    @classmethod
    @abstractmethod
    def validate(cls, module):
        pass

    @classmethod
    def gen_quant_impl_kwargs(
            cls, scale: Tensor, zero_point: Tensor, signed: bool, include_dtype=True):
        if _is_scalar(scale):
            assert _is_scalar(zero_point), 'Scalar zero point required'
            scale, zero_point = scale.item(), zero_point.item()
            quant_impl = torch.quantize_per_tensor
        else:
            if _is_scalar(zero_point):
                zero_point = zero_point.expand_as(scale)
            quant_impl = torch.quantize_per_channel
        quant_kwargs = {'scale': scale, 'zero_point': zero_point}
        if include_dtype and signed:
            quant_kwargs['dtype'] = torch.qint8
        elif include_dtype and not signed:
            quant_kwargs['dtype'] = torch.quint8
        return quant_impl, quant_kwargs

    @classmethod
    def prepare_input_quant(cls, module):
        scale = module.quant_input_scale()
        zero_point = cls.quant_input_zero_point(module)
        signed = module.is_quant_input_signed
        quant_impl, quant_kwargs = cls.gen_quant_impl_kwargs(scale, zero_point, signed)
        return quant_impl, quant_kwargs

    @classmethod
    def prepare_output_quant(cls, module):
        scale = module.quant_output_scale()
        zero_point = cls.quant_output_zero_point(module)
        signed = module.is_quant_output_signed
        incl_dtype = cls.explicit_output_dtype()
        quant_impl, quant_kwargs = cls.gen_quant_impl_kwargs(scale, zero_point, signed, incl_dtype)
        return quant_impl, quant_kwargs