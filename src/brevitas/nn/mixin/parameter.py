# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABCMeta
from abc import abstractmethod
from typing import Optional, Type, Union
from warnings import warn

from brevitas.inject import ExtendedInjector
from brevitas.inject import Injector
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import BiasQuantProxyProtocol
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyProtocol
from brevitas.quant import NoneBiasQuant
from brevitas.quant import NoneWeightQuant
from brevitas.quant_tensor import QuantTensor

from .base import QuantProxyMixin

WeightQuantType = Union[WeightQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]
BiasQuantType = Union[BiasQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]


class QuantWeightMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(self, weight_quant: Optional[WeightQuantType], **kwargs):
        QuantProxyMixin.__init__(
            self,
            quant=weight_quant,
            proxy_protocol=WeightQuantProxyProtocol,
            none_quant_injector=NoneWeightQuant,
            kwargs_prefix='weight_',
            proxy_prefix='weight_',
            **kwargs)

    @property
    @abstractmethod
    def output_channel_dim(self) -> int:
        pass

    @property
    def is_weight_quant_enabled(self):
        return self.weight_quant.is_quant_enabled

    @property
    def is_quant_weight_narrow_range(self):
        return self.weight_quant.is_narrow_range

    @property
    def is_quant_weight_signed(self):
        return self.weight_quant.is_signed

    @property
    def weight_quant_requires_quant_input(self):
        return self.weight_quant.requires_quant_input

    def quant_weight(self, quant_input: Optional[QuantTensor] = None):
        if self.weight_quant_requires_quant_input:
            if quant_input is None:
                input_bit_width = self.quant_input_bit_width()
                input_is_signed = self.is_quant_input_signed
            else:
                input_bit_width = quant_input.bit_width
                input_is_signed = quant_input.signed
            assert input_bit_width is not None, "Input bit-width needs to be specified."
            assert input_is_signed is not None, "Input sign needs to be specified."
            return self.weight_quant(self.weight, input_bit_width, input_is_signed)
        return self.weight_quant(self.weight)

    def int_weight(self, float_datatype=False):
        return self.quant_weight().int(float_datatype)

    def quant_weight_scale(self):
        scale = self.quant_weight().scale
        return scale

    def quant_weight_zero_point(self):
        scale = self.quant_weight().zero_point
        return scale

    def quant_weight_bit_width(self):
        bit_width = self.quant_weight().bit_width
        return bit_width

    def register_parameter(self, name, value):
        super(QuantWeightMixin, self).register_parameter(name, value)
        if hasattr(self, 'weight_quant') and name == 'weight':
            self.weight_quant.init_tensor_quant()
            self.weight_quant.to(self.weight.device)


class QuantBiasMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            bias_quant: Optional[BiasQuantType],
            cache_inference_bias: bool = False,
            **kwargs):
        QuantProxyMixin.__init__(
            self,
            quant=bias_quant,
            proxy_protocol=BiasQuantProxyProtocol,
            none_quant_injector=NoneBiasQuant,
            kwargs_prefix='bias_',
            proxy_prefix='bias_',
            **kwargs)
        self.cache_inference_quant_bias = cache_inference_bias
        self._cached_bias = None

    @property
    def is_bias_quant_enabled(self):
        return self.bias_quant.is_quant_enabled

    @property
    def is_quant_bias_narrow_range(self):
        if self.bias is None:
            return None
        return self.bias_quant.is_narrow_range

    @property
    def is_quant_bias_signed(self):
        if self.bias is None or not self.is_bias_quant_enabled:
            return None
        return self.bias_quant.is_signed

    def int_bias(self, float_datatype=False):
        if self.bias is None or not self.is_bias_quant_enabled:
            return None
        quant_bias = self.quant_bias()
        return quant_bias.int(float_datatype=float_datatype)

    def quant_bias(self):
        if self.bias is None:
            return None
        scale = self.quant_bias_scale()
        bit_width = self.quant_bias_bit_width()
        quant_bias = self.bias_quant(self.bias, scale, bit_width)
        return quant_bias

    def quant_bias_scale(self):
        if self.bias is None or not self.is_bias_quant_enabled:
            return None
        if not self.bias_quant.requires_input_scale and not self.bias_quant.requires_input_bit_width:
            return self.bias_quant(self.bias).scale
        else:
            if self._cached_bias is None:
                raise RuntimeError(
                    "No quant bias cache found, set cache_inference_quant_bias=True and run an "
                    "inference pass first")
            if self.training:
                warn("Cached quant bias scale is being used in training mode.")
            return self._cached_bias.scale

    def quant_bias_zero_point(self):
        if self.bias is None:
            return None
        if not self.bias_quant.requires_input_scale and not self.bias_quant.requires_input_bit_width:
            return self.bias_quant(self.bias).zero_point
        else:
            if self._cached_bias is None:
                raise RuntimeError(
                    "No quant bias cache found, set cache_inference_quant_bias=True and run an "
                    "inference pass first")
            if self.training:
                warn("Cached quant bias zero-point is being used in training mode.")
            return self._cached_bias.bit_width

    def quant_bias_bit_width(self):
        if self.bias is None or not self.is_bias_quant_enabled:
            return None
        if not self.bias_quant.requires_input_scale and not self.bias_quant.requires_input_bit_width:
            return self.bias_quant(self.bias).bit_width
        else:
            if self._cached_bias is None:
                raise RuntimeError(
                    "No quant bias cache found, set cache_inference_quant_bias=True and run an "
                    "inference pass first")
            if self.training:
                warn("Cached quant bias bit-width is being used in training mode.")
            return self._cached_bias.bit_width

    def register_parameter(self, name, value):
        super(QuantBiasMixin, self).register_parameter(name, value)
        if hasattr(self, 'bias_quant') and name == 'bias':
            self.bias_quant.init_tensor_quant()
            self.bias_quant.to(self.bias.device)
