# -*- coding: future_annotations -*-

from abc import ABC
from typing import Tuple, Union, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from brevitas.nn.mixin.base import QuantLayerMixin
    from brevitas.nn.mixin.act import QuantOutputMixin
from brevitas.export.onnx.handler import ONNXBaseHandler
from ..utils import finn_datatype


class FINNQuantInputHandler(ONNXBaseHandler, ABC):

    @staticmethod
    def quant_input_scale(module: QuantLayerMixin):
        scale = module.quant_input_scale()
        if scale is None:
            return None
        if not module.channelwise_separable:
            scale = scale.type(torch.FloatTensor).detach()
            scale = torch.tensor(scale.item())
            return scale
        else:
            scale = scale.type(torch.FloatTensor).detach()
            if scale.view(-1).shape[0] == 1:
                scale = torch.tensor(scale.item())
            return scale

    @staticmethod
    def quant_input_signed(module: QuantLayerMixin) -> Optional[bool]:  # tri-valued logic output
        signed = module.is_quant_input_signed
        return signed

    @staticmethod
    def quant_input_bit_width_tensor(module: QuantLayerMixin):
        bit_width = module.quant_input_bit_width()
        return bit_width

    @staticmethod
    def quant_input_shape(module: QuantLayerMixin):
        cached_inp = module._cached_inp
        if cached_inp is not None:
            return cached_inp.shape
        return None

    @staticmethod
    def quant_input_type(module: QuantLayerMixin):
        input_bit_width_tensor = FINNQuantInputHandler.quant_input_bit_width_tensor(module)
        input_signed = FINNQuantInputHandler.quant_input_signed(module)
        if input_bit_width_tensor is not None and input_signed is not None:
            return finn_datatype(input_bit_width_tensor, input_signed)
        else:
            return None


class FINNQuantIOHandler(FINNQuantInputHandler, ABC):

    @staticmethod
    def quant_output_scale(module: Union[QuantLayerMixin, QuantOutputMixin]):
        scale = module.quant_output_scale()
        if scale is None:
            return None
        scale = scale.type(torch.FloatTensor).detach()
        if scale.view(-1).shape[0] == 1:
            scale = torch.tensor(scale.item())
        return scale

    @staticmethod
    def quant_output_signed(
            module: Union[QuantLayerMixin, QuantOutputMixin]) -> Optional[bool]:  # tri-valued logic
        signed = module.is_quant_output_signed
        return signed

    @staticmethod
    def quant_output_bit_width_tensor(module: Union[QuantLayerMixin, QuantOutputMixin]):
        bit_width = module.quant_output_bit_width()
        return bit_width

    @staticmethod
    def quant_output_shape(module: Union[QuantLayerMixin, QuantOutputMixin]):
        cached_out = module._cached_out  # TODO add shape property to the module
        if cached_out is not None:
            return cached_out.shape
        return None