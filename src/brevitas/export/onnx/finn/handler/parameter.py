from abc import ABC
from typing import Optional, Union

import torch
from torch import Tensor

from brevitas.nn import QuantLinear, QuantConv2d, QuantConv1d
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.export.onnx.handler import Kernel2dApplHandlerMixin, Kernel1dApplHandlerMixin
from .base import FINNQuantIOHandler
from ..function.parameter import QuantizedLinearFn
from ..function.parameter import QuantizedConvNdFn
from ..utils import finn_datatype

QuantConvNd = Union[QuantConv1d, QuantConv2d]


class FINNQuantWBIOLHandler(FINNQuantIOHandler, ABC):

    @staticmethod
    def validate(module: QuantWBIOL):
        assert module.is_weight_quant_enabled
        assert not module.is_input_quant_enabled
        assert not module.is_output_quant_enabled
        if module.is_bias_quant_enabled:
            assert module.bias_quant.requires_input_scale

    @staticmethod
    def quant_weight_type(module: QuantWBIOL):
        return finn_datatype(module.quant_weight_bit_width(), module.is_quant_weight_signed)

    @staticmethod
    def int_weight(module: QuantConvNd):
        return module.int_weight(float_datatype=True).detach()

    @staticmethod
    def int_weight_transposed(module: QuantWBIOL):
        return torch.t(module.int_weight(float_datatype=True)).detach()

    @staticmethod
    def maybe_int_bias(module: QuantWBIOL):
        if module.bias is not None:
            if module.is_bias_quant_enabled:
                bias = module.int_bias(float_datatype=True)
            else:
                bias = module.bias
            bias = torch.t(bias).detach()
        else:
            bias = None
        return bias

    @staticmethod
    def maybe_quant_bias_type(module: QuantWBIOL):
        if module.is_bias_quant_enabled:
            return finn_datatype(module.quant_bias_bit_width(), module.is_quant_bias_signed)
        else:
            return None

    @staticmethod
    def maybe_quant_bias_scale(module: QuantWBIOL):
        if module.is_bias_quant_enabled:
            return module.quant_bias_scale()
        else:
            return None


class FINNQuantLinearHandler(FINNQuantWBIOLHandler):
    handled_layer = QuantLinear

    @staticmethod
    def quant_weight_scale(module: QuantWBIOL):
        return torch.t(module.quant_weight_scale().type(torch.FloatTensor)).detach()

    @staticmethod
    def quant_output_shape(module: QuantLinear):
        shape = FINNQuantWBIOLHandler.quant_output_shape(module)
        if shape is None:
            return (1, module.out_features)
        else:
            return shape

    def prepare_for_export(self, module):
        self.validate(module)
        self.symbolic_kwargs = {
            'Wt': self.int_weight_transposed(module),
            'w_qnt_scale': self.quant_weight_scale(module),
            'b_qnt_scale': self.maybe_quant_bias_scale(module),
            'w_qnt_type': self.quant_weight_type(module),
            'b_qnt_type': self.maybe_quant_bias_type(module),
            'out_shape': self.quant_output_shape(module),
            'bias': self.maybe_int_bias(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = QuantizedLinearFn.apply(inp, *self.symbolic_kwargs.values())
        return ret


class FINNQuantConvNdHandler(FINNQuantWBIOLHandler, ABC):

    @staticmethod
    def quant_output_shape(module: QuantConvNd):
        shape = FINNQuantWBIOLHandler.quant_output_shape(module)
        if shape is None:
            raise RuntimeError("Caching of output shapes is required to export QuantConvNd")
        return shape

    @staticmethod
    def maybe_int_bias(module: QuantWBIOL):
        if module.bias is not None:
            if module.is_bias_quant_enabled:
                bias = module.int_bias(float_datatype=True)
            else:
                bias = module.bias
            bias_shape = [1] * len(module.weight.shape)
            bias_shape[1] = -1
            # shape should broadcast with activations along channel dim
            bias = bias.view(bias_shape).detach()
        else:
            bias = None
        return bias

    def prepare_for_export(self, module: QuantConvNd):
        self.validate(module)
        maybe_int_bias = self.maybe_int_bias(module)
        maybe_quant_bias_scale = self.maybe_quant_bias_scale(module)
        if (maybe_quant_bias_scale is not None
                and len(maybe_quant_bias_scale.shape) > 0
                and len(maybe_quant_bias_scale.view(-1)) > 1):
            maybe_quant_bias_scale = maybe_quant_bias_scale.view_as(maybe_int_bias)
        self.symbolic_kwargs = {
            'W': self.int_weight(module),
            'w_qnt_scale': self.quant_weight_scale(module),
            'b_qnt_scale': maybe_quant_bias_scale,
            'w_qnt_type': self.quant_weight_type(module),
            'b_qnt_type': self.maybe_quant_bias_type(module),
            'out_shape': self.quant_output_shape(module),
            'pads': self.padding(module),
            'strides': self.stride(module),
            'bias': maybe_int_bias,
            'kernel_shape': list(module.kernel_size),
            'groups': module.groups,
            'dilations': self.dilation(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = QuantizedConvNdFn.apply(inp, *self.symbolic_kwargs.values())
        return ret


class FINNQuantConv1dHandler(FINNQuantConvNdHandler, Kernel1dApplHandlerMixin):
    handled_layer = QuantConv1d

    @staticmethod
    def quant_weight_scale(module: QuantConv1d):
        quant_weight_scale = module.quant_weight_scale().type(torch.FloatTensor).detach()
        if len(quant_weight_scale.shape) == 3:
            quant_weight_scale = quant_weight_scale.view(1, -1, 1)
        return quant_weight_scale


class FINNQuantConv2dHandler(FINNQuantConvNdHandler, Kernel2dApplHandlerMixin):
    handled_layer = QuantConv2d

    @staticmethod
    def quant_weight_scale(module: QuantConv2d):
        quant_weight_scale = module.quant_weight_scale().type(torch.FloatTensor).detach()
        if len(quant_weight_scale.shape) == 4:
            quant_weight_scale = quant_weight_scale.view(1, -1, 1, 1)
        return quant_weight_scale
