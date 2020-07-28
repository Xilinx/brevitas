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

from abc import ABCMeta, abstractmethod
from typing import Optional, Type, Union, Callable
import torch

from dependencies import Injector

from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector, BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyProtocol, BiasQuantProxyProtocol
from brevitas.proxy.parameter_quant import ParameterQuantProxyFromInjector
from brevitas.proxy.parameter_quant import ParameterQuantProxyProtocol

class QuantParameterMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            parameter: torch.nn.Parameter,
            parameter_quant: Optional[Union[ParameterQuantProxyProtocol, Type[Injector]]],
            proxy_from_injector_impl: Optional[Type[ParameterQuantProxyFromInjector]],
            update_injector: Optional[Callable],
            prefix: str,
            **kwargs):

        def update_pqi(pqi):
            if update_injector is not None:
                return update_injector(self, pqi, prefix, **kwargs)
            else:
                return pqi

        proxy_name = prefix + 'quant'
        if parameter_quant is None:
            assert proxy_from_injector_impl is not None
            parameter_quant_injector = Injector.let(tensor_quant=None)
            parameter_quant_injector = update_pqi(parameter_quant_injector)
            parameter_quant = proxy_from_injector_impl(parameter_quant_injector)
        elif issubclass(parameter_quant, Injector):
            assert proxy_from_injector_impl is not None
            parameter_quant_injector = parameter_quant
            parameter_quant_injector = update_pqi(parameter_quant_injector)
            parameter_quant = proxy_from_injector_impl(parameter_quant_injector)
        else:
            assert isinstance(parameter_quant, ParameterQuantProxyProtocol)
        setattr(self, proxy_name, parameter_quant)
        getattr(self, proxy_name).add_tracked_parameter(parameter)


class QuantWeightMixin(QuantParameterMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight: torch.nn.Parameter,
            weight_quant: Optional[Union[WeightQuantProxyProtocol, Type[Injector]]],
            update_injector: Optional[Callable],
            **kwargs):
        QuantParameterMixin.__init__(
            self,
            parameter=weight,
            parameter_quant=weight_quant,
            proxy_from_injector_impl=WeightQuantProxyFromInjector,
            update_injector=update_injector,
            prefix='weight_',
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
        assert self.is_weight_quant_enabled, "Weight quantization disabled"
        return self.weight_quant.is_narrow_range

    @property
    def is_quant_weight_signed(self):
        assert self.is_weight_quant_enabled
        return self.weight_quant.is_signed

    def quant_weight(self):
        return self.weight_quant(self.weight)

    def int_weight(self, float_datatype=False):
        assert self.is_weight_quant_enabled, "Weight quantization disabled"
        return self.quant_weight().int(float_datatype)

    def quant_weight_scale(self):
        assert self.is_weight_quant_enabled, "Weight quantization disabled"
        scale = self.quant_weight().scale
        return scale

    def quant_weight_bit_width(self):
        assert self.is_weight_quant_enabled, "Weight quantization disabled"
        bit_width = self.quant_weight().bit_width
        return bit_width


class QuantBiasMixin(QuantParameterMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            bias: torch.nn.Parameter,
            bias_quant: Union[BiasQuantProxyProtocol, Type[Injector]],
            update_injector: Callable,
            **kwargs):
        QuantParameterMixin.__init__(
            self,
            parameter=bias,
            parameter_quant=bias_quant,
            proxy_from_injector_impl=BiasQuantProxyFromInjector,
            update_injector=update_injector,
            prefix='bias_',
            **kwargs)

    @property
    def is_bias_quant_enabled(self):
        return self.bias_quant.is_quant_enabled

    @property
    def is_quant_bias_narrow_range(self):
        assert self.is_bias_quant_enabled, "Bias quantization disabled"
        return self.weight_quant.is_narrow_range

    @property
    def is_quant_bias_signed(self):
        assert self.is_bias_quant_enabled, "Bias quantization disabled"
        return self.weight_quant.is_signed