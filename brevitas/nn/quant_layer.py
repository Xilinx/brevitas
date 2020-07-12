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
from torch import Tensor
from torch.nn import Identity, Module

from dependencies import Injector

from brevitas.quant_tensor import QuantTensor
from brevitas.proxy.parameter_quant import ParameterQuantProxy, WeightQuantProxy, BiasQuantProxy
from brevitas.proxy.runtime_quant import ActivationQuantProxy


def _compute_channel_view_shape(tensor, channel_dim=1):
    broadcast_shape = [1] * len(tensor.size())
    broadcast_shape[channel_dim] = -1
    return tuple(broadcast_shape)


class QuantLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, return_quant_tensor):
        self.return_quant_tensor = return_quant_tensor

    def unpack_input(self, inp):
        if isinstance(inp, QuantTensor):
            return inp
        else:
            return QuantTensor(inp)

    def pack_output(self, quant_output: QuantTensor):
        if self.return_quant_tensor:
            return quant_output
        else:
            return quant_output.value


class QuantParameterMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            parameter: torch.nn.Parameter,
            parameter_quant: Optional[Union[ParameterQuantProxy, Type[Injector]]],
            proxy_impl: Optional[Type[ParameterQuantProxy]],
            update_injector: Optional[Callable],
            prefix: str,
            **kwargs):
        self.quant_attr_name = prefix + 'quant'
        if parameter_quant is None:
            assert proxy_impl is not None
            none_injector = proxy_impl(Injector.let(tensor_quant=None))
            setattr(self, self.quant_attr_name, none_injector)
        elif isinstance(parameter_quant, ParameterQuantProxy):
            pass
        else:
            assert proxy_impl is not None
            parameter_quant_injector = parameter_quant
            if update_injector is not None:
                parameter_quant_injector = update_injector(
                    parameter_layer=self,
                    parameter_quant_injector=parameter_quant_injector,
                    prefix=prefix,
                    **kwargs)
            parameter_quant = proxy_impl(parameter_quant_injector)
        setattr(self, self.quant_attr_name, parameter_quant)
        getattr(self, self.quant_attr_name).add_tracked_parameter(parameter)


class QuantWeightMixin(QuantParameterMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight: torch.nn.Parameter,
            weight_quant: Optional[Union[WeightQuantProxy, Type[Injector]]],
            update_injector: Optional[Callable],
            **kwargs):
        super().__init__(
            parameter=weight,
            parameter_quant=weight_quant,
            proxy_impl=WeightQuantProxy,
            update_injector=update_injector,
            prefix='weight_',
            **kwargs)

    @property
    @abstractmethod
    def output_channel_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def channelwise_separable(self) -> bool:
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        pass

    def quant_weight(self):
        return self.weight_quant(self.weight)

    def int_weight(self, float_datatype=False):
        return self.quant_weight().int(float_datatype)

    def quant_weight_scale(self):
        scale = self.quant_weight().scale
        return scale


class QuantBiasMixin(QuantParameterMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            bias,
            bias_quant: Union[BiasQuantProxy, Type[Injector]],
            update_injector: Callable,
            **kwargs):
        super().__init__(
            parameter=bias,
            parameter_quant=bias_quant,
            proxy_impl=BiasQuantProxy,
            update_injector=update_injector,
            prefix='bias_',
            **kwargs)


class QuantActivationMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            activation_impl: Module,
            activation_quant: Union[ActivationQuantProxy, Type[Injector]],
            update_injector: Callable,
            prefix: str = 'activation_',
            **kwargs):
        attr_name = prefix + 'quant'
        if activation_quant is None:
            activation_quant_injector = Injector.let(tensor_quant=None)
            activation_quant = ActivationQuantProxy(activation_impl, activation_quant_injector)
        elif isinstance(activation_quant, ActivationQuantProxy):
            pass
        else:
            activation_quant_injector = activation_quant
            if update_injector is not None:
                activation_quant_injector = update_injector(
                    activation_layer=self,
                    activation_quant_injector=activation_quant_injector,
                    prefix=prefix,
                    **kwargs)
            activation_quant = ActivationQuantProxy(Identity(), activation_quant_injector)
        setattr(self, attr_name, activation_quant)


class QuantInputMixin(QuantActivationMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            activation_quant: Union[ActivationQuantProxy, Type[Injector]],
            update_injector: Callable,
            **kwargs):
        super().__init__(
            Identity(),
            activation_quant,
            update_injector,
            prefix='input_',
            **kwargs)


class QuantOutputMixin(QuantActivationMixin):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            activation_quant: Union[ActivationQuantProxy, Type[Injector]],
            update_injector: Callable,
            **kwargs):
        super().__init__(
            Identity(),
            activation_quant,
            update_injector,
            prefix='output_',
            **kwargs)


class QuantWeightBiasInputOutputLayer(
        QuantOutputMixin,
        QuantInputMixin,
        QuantBiasMixin,
        QuantWeightMixin,
        QuantLayer):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight,
            weight_quant: Union[WeightQuantProxy, Type[Injector]],
            bias_quant: Union[BiasQuantProxy, Type[Injector]],
            input_quant: Union[ActivationQuantProxy, Type[Injector]],
            output_quant: Union[ActivationQuantProxy, Type[Injector]],
            update_wqi: Callable,
            update_bqi: Callable,
            update_iqi: Callable,
            update_oqi: Callable,
            return_quant_tensor: bool,
            **kwargs):
        QuantLayer.__init__(self, return_quant_tensor)
        QuantWeightMixin.__init__(self, weight, weight_quant, update_wqi, **kwargs)
        QuantBiasMixin.__init__(self, bias_quant, update_bqi, **kwargs)
        QuantInputMixin.__init__(self, input_quant, update_iqi, **kwargs)
        QuantOutputMixin.__init__(self, output_quant, update_oqi, **kwargs)

    @abstractmethod
    def max_acc_bit_width(self, input_bit_width: Tensor, quant_weight_bit_width: Tensor):
        pass

    @abstractmethod
    def inner_forward_impl(
            self, x: QuantTensor, quant_weight: QuantTensor, quant_bias: Optional[QuantTensor]):
        pass

    def forward(self, inp: Union[Tensor, QuantTensor]):
        output_scale = None
        output_bit_width = None

        inp = self.unpack_input(inp)
        quant_input = self.input_quant(inp)
        quant_weight = self.weight_quant(self.weight)

        if quant_input.bit_width is not None:
            output_bit_width = self.max_acc_bit_width(quant_input.bit_width, quant_weight.bit_width)
        if quant_input.scale is not None:
            output_scale_shape = _compute_channel_view_shape(inp)
            output_scale = quant_weight.scale.view(output_scale_shape)
            output_scale = output_scale * quant_input.scale.view(output_scale_shape)

        if self.bias is not None:
            quant_bias = self.bias_quant(self.bias, output_scale, output_bit_width)
            output_tensor = self.inner_forward_impl(quant_input, quant_weight, quant_bias)
            if quant_bias.bit_width is not None:
                output_bit_width = torch.where(
                    quant_bias.bit_width > output_bit_width, quant_bias.bit_width, output_bit_width)
                output_bit_width = output_bit_width + 1
        else:
            output_tensor = self.inner_forward_impl(quant_input, quant_weight, None)

        quant_output = QuantTensor(output_tensor, output_scale, output_bit_width, signed=True)
        quant_output = self.output_quant(quant_output)
        return self.pack_output(quant_output)









