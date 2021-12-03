# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from warnings import warn
from abc import ABCMeta, abstractmethod
from typing import Optional, Type, Union

from brevitas.inject import ExtendedInjector, Injector
from brevitas.quant import NoneWeightQuant, NoneBiasQuant
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector, BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyProtocol, BiasQuantProxyProtocol

from .base import QuantProxyMixin


WeightQuantType = Union[WeightQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]
BiasQuantType = Union[BiasQuantProxyProtocol, Type[Injector], Type[ExtendedInjector]]


class QuantWeightMixin(QuantProxyMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight_quant: Optional[WeightQuantType],
            **kwargs):
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

    def quant_weight(self):
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
            proxy_from_injector_impl=BiasQuantProxyFromInjector,
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

