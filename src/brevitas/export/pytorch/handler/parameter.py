import warnings
from typing import Union
from abc import abstractmethod, ABC

import torch
from torch import Tensor
from torch.nn.quantized import functional as qF

from brevitas.nn import QuantConv2d, QuantConv1d, QuantLinear
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL

from .base import PytorchQuantLayerHandler


class PytorchQuantWBIOLHandler(PytorchQuantLayerHandler):

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.qf_impl = None
        self.input_quant_impl = None
        self.weight_quant_impl = None
        self.weight_quant_args = None
        self.input_quant_kwargs = None
        self.weight_quant_kwargs = None
        self.output_quant_kwargs = None
        self.qf_kwargs = None
        self.return_quant_tensor = None

    @classmethod
    def validate(cls, module: QuantWBIOL):
        assert module.is_weight_quant_enabled, 'Weight quantization required'
        assert module.is_output_quant_enabled, 'Output quantization required'
        if module.is_bias_quant_enabled:
            warnings.warn('Bias quantization not supported, it will be ignored')

    @classmethod
    def prepare_weight_quant(cls, module: QuantWBIOL):
        cls.validate_8b_bit_width(module.quant_weight_bit_width())
        scale = module.quant_weight_scale()
        zero_point = cls.quant_weight_zero_point(module)
        signed = module.is_quant_weight_signed
        weight = module.weight.detach()
        quant_impl, quant_kwargs = cls.gen_quant_impl_kwargs(scale, zero_point, signed)
        return quant_impl, (weight,), quant_kwargs

    def prepare_for_export(self, module: QuantWBIOL):
        self.validate(module)
        if module.is_input_quant_enabled:
            self.input_quant_impl, self.input_quant_kwargs = self.prepare_input_quant(module)
        weight_quant_pack = self.prepare_weight_quant(module)
        self.weight_quant_impl, self.weight_quant_args, self.weight_quant_kwargs = weight_quant_pack
        self.qf_impl, self.qf_kwargs = self.prepare_qf(module)
        _, self.output_quant_kwargs = self.prepare_output_quant(module)
        self.return_quant_tensor = module.return_quant_tensor

    def q_weight(self):
        q_weight = self.weight_quant_impl(*self.weight_quant_args, **self.weight_quant_kwargs)
        return q_weight

    def forward(self, q_inp: Tensor):
        if self.input_quant_impl is not None:
            q_inp = self.input_quant_impl(q_inp, **self.input_quant_kwargs)
        assert q_inp.is_quantized, 'Input needs to be quantized'
        q_out = self.qf_impl(q_inp, self.q_weight(), **self.qf_kwargs, **self.output_quant_kwargs)
        if not self.return_quant_tensor:
            q_out = q_out.dequantize()
        return q_out


class PytorchQuantConvNdHandler(PytorchQuantWBIOLHandler, ABC):

    @classmethod
    def explicit_output_dtype(cls):
        return True

    @classmethod
    def prepare_qf_kwargs(self, module: Union[QuantConv1d, QuantConv2d]):
        return {
            'bias': module.bias,
            'stride': module.stride,
            'padding': module.padding,
            'dilation': module.dilation,
            'groups': module.groups,
            'padding_mode': module.padding_mode}


class PytorchQuantConv1dHandler(PytorchQuantConvNdHandler):
    handled_layer = QuantConv1d

    @classmethod
    def prepare_qf(cls, module: QuantConv1d):
        return qF.conv1d, cls.prepare_qf_kwargs(module)


class PytorchQuantConv2dHandler(PytorchQuantConvNdHandler):
    handled_layer = QuantConv2d

    @classmethod
    def prepare_qf(cls, module: QuantConv2d):
        return qF.conv2d, cls.prepare_qf_kwargs(module)


class PytorchQuantLinearHandler(PytorchQuantWBIOLHandler):
    handled_layer = QuantLinear

    @classmethod
    def explicit_output_dtype(cls):
        return False

    @classmethod
    def prepare_qf(cls, module: QuantLinear):
        return qF.linear, {'bias': module.bias}


