from abc import ABC
from typing import Optional, Union

import torch
from torch import Tensor

from brevitas.nn import QuantLinear, QuantConv2d, QuantConv1d
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.export.onnx.handler import Kernel2dApplHandler, Kernel1dApplHandler
from .base import FINNQuantIOHandler
from ..function.parameter import QuantizedLinearPlaceholderFunction
from ..function.parameter import QuantizedConvNdPlaceholderFunction


QuantConvNd = Union[QuantConv1d, QuantConv2d]


class FINNQuantWBIOLHandler(FINNQuantIOHandler, ABC):

    @staticmethod
    def sanity_check(module: QuantWBIOL):
        assert module.is_weight_quant_enabled
        assert not module.is_input_quant_enabled
        assert not module.is_output_quant_enabled

    @staticmethod
    def quant_weight_type(module: QuantWBIOL):
        bit_width = int(module.quant_weight_bit_width().item())
        if bit_width == 1:
            return "BIPOLAR"
        else:
            return f"INT{bit_width}"

    @staticmethod
    def function_scaling(
            module: QuantWBIOL,
            quant_input_scale,
            quant_weight_scale,
            quant_output_scale):
        # if bias quant is disabled, whether bias is enabled or not, I just need to
        # pass quant_weight_scale as the function-level scale_factor
        if not module.is_bias_quant_enabled:
            in_scale, scale_factor = None, quant_weight_scale
        else:
            in_scale, scale_factor = quant_input_scale, quant_output_scale
        return in_scale, scale_factor

    @staticmethod
    def maybe_quant_bias(
            module: QuantWBIOL,
            quant_weight_scale: Tensor,
            quant_output_scale: Optional[Union[Tensor, float]],
            quant_bit_width: Optional[Tensor]):
        if module.bias is None:
            return None
        elif module.bias is not None and not module.is_bias_quant_enabled:
            bias = torch.t(module.bias.type(torch.FloatTensor)).detach()
            # divide by weight scale as add is before mul
            bias /= quant_weight_scale
            return bias
        else:  # bias quant enabled
            assert quant_output_scale, 'Quant bias export requires caching of the output scale'
            if not isinstance(quant_output_scale, Tensor):  # item might have been called
                quant_output_scale = torch.tensor(quant_output_scale)
            quant_bias = module.bias_quant(module.bias, quant_output_scale, quant_bit_width)
            quant_bias = torch.t(quant_bias.value.type(torch.FloatTensor)).detach()
            quant_bias /= quant_output_scale
            quant_bias = torch.round(quant_bias)
            return quant_bias


class FINNQuantLinearHandler(FINNQuantWBIOLHandler):
    handled_layer = QuantLinear

    @staticmethod
    def int_weight(module: QuantWBIOL):
        return torch.t(module.int_weight(float_datatype=True)).detach()

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
        self.sanity_check(module)
        quant_input_scale = self.quant_input_scale(module)
        quant_weight_scale = self.quant_weight_scale(module)
        quant_output_scale = self.quant_output_scale(module)
        quant_output_bit_width_tensor = self.quant_output_bit_width_tensor(module)
        maybe_quant_bias = self.maybe_quant_bias(
            module,
            quant_weight_scale,
            quant_output_scale,
            quant_output_bit_width_tensor)
        in_scale, scale_factor = self.function_scaling(
            module, quant_input_scale, quant_weight_scale, quant_output_scale)
        self.symbolic_kwargs = {
            'Wt': self.int_weight(module),
            'scale_factor': scale_factor,
            'w_qnt_type': self.quant_weight_type(module),
            'out_shape': self.quant_output_shape(module),
            'bias': maybe_quant_bias,
            'in_scale': in_scale,
            'in_qnt_type': self.quant_input_type(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = QuantizedLinearPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class FINNQuantConvNdHandler(FINNQuantWBIOLHandler, ABC):

    @staticmethod
    def int_weight(module: QuantConvNd):
        return module.int_weight(float_datatype=True).detach()

    @staticmethod
    def quant_output_shape(module: QuantConvNd):
        shape = FINNQuantWBIOLHandler.quant_output_shape(module)
        if shape is None:
            raise RuntimeError("Caching of output shapes is required to export QuantConvNd")
        return shape

    @staticmethod
    def maybe_quant_bias(
            module: QuantWBIOL,
            quant_weight_scale: Tensor,
            quant_output_scale: Optional[Union[Tensor, float]],
            quant_bit_width: Optional[Tensor]):
        quant_bias = FINNQuantWBIOLHandler.maybe_quant_bias(
            module, quant_weight_scale, quant_output_scale, quant_bit_width)
        if quant_bias is not None:
            quant_bias_shape = [1] * len(module.weight.shape)
            quant_bias_shape[1] = -1 # shape should broadcast with activations along channel dim
            quant_bias = quant_bias.view(quant_bias_shape)
        return quant_bias

    def prepare_for_export(self, module: QuantConvNd):
        self.sanity_check(module)
        quant_input_scale = self.quant_input_scale(module)
        quant_weight_scale = self.quant_weight_scale(module)
        quant_output_scale = self.quant_output_scale(module)
        quant_output_bit_width_tensor = self.quant_output_bit_width_tensor(module)
        maybe_quant_bias = self.maybe_quant_bias(
            module,
            quant_weight_scale,
            quant_output_scale,
            quant_output_bit_width_tensor)
        in_scale, scale_factor = self.function_scaling(
            module, quant_input_scale, quant_weight_scale, quant_output_scale)
        self.symbolic_kwargs = {
            'W': self.int_weight(module),
            'scale_factor': scale_factor,
            'w_qnt_type': self.quant_weight_type(module),
            'out_shape': self.quant_output_shape(module),
            'pads': self.padding(module),
            'strides': self.stride(module),
            'bias': maybe_quant_bias,
            'in_scale': in_scale,
            'in_qnt_type': self.quant_input_type(module),
            'kernel_shape': list(module.kernel_size),
            'groups': module.groups,
            'dilations': self.dilation(module)}

    def symbolic_execution(self, inp: Tensor):
        ret = QuantizedConvNdPlaceholderFunction.apply(inp, *self.symbolic_kwargs.values())
        return ret


class FINNQuantConv1dHandler(FINNQuantConvNdHandler, Kernel1dApplHandler):
    handled_layer = QuantConv1d

    @staticmethod
    def quant_weight_scale(module: QuantConv1d):
        quant_weight_scale = module.quant_weight_scale().type(torch.FloatTensor).detach()
        if len(quant_weight_scale.shape) == 3:
            quant_weight_scale = quant_weight_scale.view(1, -1, 1)
        return quant_weight_scale


class FINNQuantConv2dHandler(FINNQuantConvNdHandler, Kernel2dApplHandler):
    handled_layer = QuantConv2d

    @staticmethod
    def quant_weight_scale(module: QuantConv2d):
        quant_weight_scale = module.quant_weight_scale().type(torch.FloatTensor).detach()
        if len(quant_weight_scale.shape) == 4:
            quant_weight_scale = quant_weight_scale.view(1, -1, 1, 1)
        return quant_weight_scale
