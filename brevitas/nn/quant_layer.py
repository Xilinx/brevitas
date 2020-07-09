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
from dependencies import Injector

import torch
from torch import Tensor
from torch.nn import Identity

from brevitas.quant_tensor import QuantTensor
from brevitas.core.quant import IdentityQuant, IdentityPrescaledIntQuant
from brevitas.proxy import WeightQuantProxy, BiasQuantProxy, ActivationQuantProxy


OVER_BATCH_OVER_CHANNELS_4D_SHAPE = (1, -1, 1, 1)


class QuantLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, return_quant_tensor):
        self.return_quant_tensor = return_quant_tensor

    def unpack_input(self, input):
        if isinstance(input, QuantTensor):
            return input
        else:
            return input, None, None

    def pack_output(self, quant_output: QuantTensor):
        if self.return_quant_tensor:
            return quant_output
        else:
            return quant_output.tensor

    @property
    @abstractmethod
    def returned_scale_shape(self):
        pass


class QuantWeightMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight: torch.nn.Parameter,
            weight_quant: Optional[Union[WeightQuantProxy, Type[Injector]]],
            update_inj: Optional[Callable],
            prefix: str = 'weight_',
            **kwargs):
        if weight_quant is None:
            self.weight_quant = WeightQuantProxy(weight_quant)
        elif isinstance(weight_quant, WeightQuantProxy):
            self.weight_quant = weight_quant
            self.weight_quant.add_tracked_tensor(weight)
        else:
            weight_quant_inj = weight_quant
            if update_inj is not None:
                weight_quant_inj = update_inj(
                    weight_layer=self, quant_weight_inj=weight_quant_inj, prefix=prefix, **kwargs)
            weight_quant_inj = weight_quant_inj.let(tracked_parameter_list=[weight])
            self.weight_quant = WeightQuantProxy(weight_quant_inj)

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

    def int_weight(self):
        if isinstance(self.weight_quant.tensor_quant, IdentityQuant):
            raise Exception("Can't generate int weight without quantization enabled")
        return self.weight_quant.int_weight(self.weight)

    def quant_weight_scale(self):
        if isinstance(self.weight_quant.tensor_quant, IdentityQuant):
            raise Exception("Can't generate scaling factor without quantization enabled")
        _, scale, _ = self.weight_quant.tensor_quant(self.weight)
        return scale


class QuantBiasMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            bias_quant: Union[BiasQuantProxy, Type[Injector]],
            update_inj: Callable,
            **kwargs):
        if bias_quant is None:
            self.bias_quant = IdentityPrescaledIntQuant()
        elif isinstance(bias_quant, BiasQuantProxy):
            self.bias_quant = bias_quant
        else:
            bias_quant_inj = bias_quant
            if update_inj is not None:
                bias_quant_inj = update_inj(
                    bias_layer=self, bias_quant_inj=bias_quant_inj, **kwargs)
            self.bias_quant = BiasQuantProxy(bias_quant_inj)


class QuantOutputMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            output_quant: Union[ActivationQuantProxy, Type[Injector]],
            update_inj: Callable,
            **kwargs):
        if output_quant is None:
            self.output_quant = IdentityQuant()
        elif isinstance(output_quant, ActivationQuantProxy):
            self.output_quant = output_quant
        else:
            output_quant_inj = output_quant
            if update_inj is not None:
                output_quant_inj = update_inj(
                    output_layer=self, output_quant_inj=output_quant_inj, **kwargs)
            self.output_quant = ActivationQuantProxy(Identity(), output_quant_inj)


class QuantWeightBiasOutputLayer(QuantOutputMixin, QuantBiasMixin, QuantWeightMixin, QuantLayer):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight,
            weight_quant: Union[WeightQuantProxy, Type[Injector]],
            bias_quant: Union[BiasQuantProxy, Type[Injector]],
            output_quant: Union[ActivationQuantProxy, Type[Injector]],
            update_weight_quant_inj: Callable,
            update_bias_quant_inj: Callable,
            update_output_quant_inj: Callable,
            return_quant_tensor: bool,
            **kwargs):
        QuantLayer.__init__(self, return_quant_tensor)
        QuantWeightMixin.__init__(self, weight, weight_quant, update_weight_quant_inj, **kwargs)
        QuantBiasMixin.__init__(self, bias_quant, update_bias_quant_inj, **kwargs)
        QuantOutputMixin.__init__(self, output_quant, update_output_quant_inj, **kwargs)

    @abstractmethod
    def max_acc_bit_width(self, input_bit_width: Tensor, quant_weight_bit_width: Tensor):
        pass

    @abstractmethod
    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        pass

    def forward(self, input_tensor: Union[Tensor, QuantTensor]):
        output_scale = None
        output_bit_width = None

        quant_input = self.pack_input(input_tensor)
        quant_weight = self.weight_quant(self.weight)

        if quant_input.bit_width is not None:
            output_bit_width = self.max_acc_bit_width(quant_input.bit_width, quant_weight.bit_width)
        if quant_input.scale is not None:
            output_scale = quant_input.scale * quant_weight.scale

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









