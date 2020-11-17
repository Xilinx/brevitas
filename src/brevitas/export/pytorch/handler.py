import warnings
from typing import Union
from abc import abstractmethod, ABC

import torch
from torch import Tensor
from torch.nn.quantized import functional as qF

from brevitas.nn import QuantConv2d, QuantConv1d, QuantLinear
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from ..base import BaseHandler
from ..common.handler import Validate8BitHandler, TypedZeroPointHandler

SCALAR_SHAPE = ()


def _is_scalar(x: Tensor):
    return x.shape == SCALAR_SHAPE


class PytorchQuantWBIOLHandler(Validate8BitHandler, TypedZeroPointHandler, BaseHandler, ABC):

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

    @property
    @abstractmethod
    def explicit_output_dtype(self):
        pass

    @abstractmethod
    def prepare_qf(self, module: QuantWBIOL):
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

    def prepare_input_quant(self, module: QuantWBIOL):
        self.validate_8b_bit_width(module.quant_input_bit_width())
        scale = module.quant_input_scale()
        zero_point = self.quant_input_zero_point(module)
        signed = module.is_quant_input_signed
        quant_impl, quant_kwargs = self.gen_quant_impl_kwargs(scale, zero_point, signed)
        self.input_quant_impl = quant_impl
        self.input_quant_kwargs = quant_kwargs

    def prepare_weight_quant(self, module: QuantWBIOL):
        self.validate_8b_bit_width(module.quant_weight_bit_width())
        scale = module.quant_input_scale()
        zero_point = self.quant_input_zero_point(module)
        signed = module.is_quant_weight_signed
        weight = module.weight.view(module.weight.shape)  # Parameter to Tensor
        quant_impl, quant_kwargs = self.gen_quant_impl_kwargs(scale, zero_point, signed)
        self.weight_quant_impl = quant_impl
        self.weight_quant_args = (weight,)
        self.weight_quant_kwargs = quant_kwargs

    def prepare_output_quant(self, module: QuantWBIOL):
        self.validate_8b_bit_width(module.quant_output_bit_width())
        scale = module.quant_output_scale()
        zero_point = self.quant_output_zero_point(module)
        signed = module.is_quant_output_signed
        _, quant_kwargs = self.gen_quant_impl_kwargs(
            scale, zero_point, signed, self.explicit_output_dtype)
        self.output_quant_kwargs = quant_kwargs

    def prepare_for_export(self, module: QuantConv2d):
        assert module.is_weight_quant_enabled, 'Weight quantization required'
        assert module.is_output_quant_enabled, 'Output quantization required'
        if module.is_bias_quant_enabled:
            warnings.warn('Bias quantization not supported, it will be ignored')
        self.prepare_weight_quant(module)
        self.prepare_output_quant(module)
        self.prepare_qf(module)
        if module.is_input_quant_enabled:
            self.prepare_input_quant(module)

    def forward_qf(self, q_inp: Tensor, q_weight: Tensor):
        out = self.qf_impl(q_inp, q_weight, **self.qf_kwargs, **self.output_quant_kwargs)
        return out

    def q_weight(self):
        q_weight = self.weight_quant_impl(*self.weight_quant_args, **self.weight_quant_kwargs)
        return q_weight

    def forward(self, q_inp: Tensor, **kwargs):
        if self.input_quant_impl is not None:
            q_inp = self.input_quant_impl(q_inp, **self.input_quant_kwargs)
        assert q_inp.is_quantized, 'Input needs to be quantized'
        q_out = self.forward_qf(q_inp, self.q_weight())
        return q_out


class PytorchQuantConvNdHandler(PytorchQuantWBIOLHandler, ABC):

    @property
    def explicit_output_dtype(self):
        return True

    def prepare_qf(self, module: Union[QuantConv1d, QuantConv2d]):
        self.qf_kwargs = {
            'bias': module.bias,
            'stride': module.stride,
            'padding': module.padding,
            'dilation': module.dilation,
            'groups': module.groups,
            'padding_mode': module.padding_mode}


class PytorchQuantConv1dHandler(PytorchQuantConvNdHandler):
    handled_layer = QuantConv1d

    def prepare_qf(self, module: QuantConv1d):
        super().prepare_qf(module)
        self.qf_impl = qF.conv1d


class PytorchQuantConv2dHandler(PytorchQuantConvNdHandler):
    handled_layer = QuantConv2d

    def prepare_qf(self, module: QuantConv2d):
        super().prepare_qf(module)
        self.qf_impl = qF.conv2d


class PytorchQuantLinearHandler(PytorchQuantWBIOLHandler):
    handled_layer = QuantLinear

    @property
    def explicit_output_dtype(self):
        return False

    def prepare_qf(self, module: QuantLinear):
        self.qf_impl = qF.linear
        self.qf_kwargs = {
            'bias': module.bias}


