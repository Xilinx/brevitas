from abc import ABC

import torch
from torch import Tensor

from brevitas.nn import QuantReLU, QuantIdentity, QuantHardTanh
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL

from .base import PytorchQuantLayerHandler
from . import qF


class PytorchQuantNLALHandler(PytorchQuantLayerHandler, ABC):

    @classmethod
    def explicit_output_dtype(cls) -> bool:
        return True

    @classmethod
    def validate(cls, module: QuantNLAL):
        assert not module.is_input_quant_enabled, 'Input quantization not supported'

    def prepare_for_export(self, module: QuantNLAL):
        self.validate(module)
        self.qf_impl, self.qf_kwargs = self.prepare_qf(module)
        if module.is_act_quant_enabled:
            self.output_quant_impl, self.output_quant_kwargs = self.prepare_output_quant(module)
        self.return_quant_tensor = module.return_quant_tensor

    def forward(self, inp: Tensor, **kwargs):
        q_out = inp
        if self.qf_impl is not None:
            q_out = self.qf_impl(q_out, **self.qf_kwargs)
        if self.output_quant_impl is not None:
            if q_out.is_quantized:
                q_out = q_out.dequantize()
            q_out = self.output_quant_impl(q_out, **self.output_quant_kwargs)
        if not self.return_quant_tensor:
            q_out = q_out.dequantize()
        return q_out


class PytorchQuantReLUHandler(PytorchQuantNLALHandler):
    handled_layer = QuantReLU

    @classmethod
    def prepare_qf(cls, module):
        return torch.nn.functional.relu, {}


class PytorchQuantIdentityHandler(PytorchQuantNLALHandler):
    handled_layer = QuantIdentity

    @classmethod
    def prepare_qf(cls, module):
        return None, None


class PytorchQuantHardTanhHandler(PytorchQuantIdentityHandler):
    handled_layer = QuantHardTanh