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
from typing import Optional, Type, Union

import torch
from torch import Tensor
from torch.nn import Identity, Module

from brevitas.quant_tensor import QuantTensor
from brevitas.core.quant import IdentityQuant
from brevitas.proxy.parameter_quant import WeightQuantProxy, BiasQuantProxy
from brevitas.proxy.runtime_quant import ActivationQuantProxy

from .config import ActQuantConfig, WeightQuantConfig, BiasQuantConfig

OVER_BATCH_OVER_CHANNELS_4D_SHAPE = (1, -1, 1, 1)


def filter_kwargs(kwargs_prefix, kwargs: dict):
    return {k[len(k):]: v for (k, v) in kwargs.items() if k.startswith(kwargs_prefix)}


class QuantLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, return_quant_tensor):
        self.return_quant_tensor = return_quant_tensor

    def unpack_input(self, input):
        if isinstance(input, QuantTensor):
            return input
        else:
            return input, None, None

    def pack_output(self,
                    output,
                    output_scale,
                    output_bit_width):
        if self.return_quant_tensor:
            return QuantTensor(
                tensor=output, scale=output_scale, bit_width=output_bit_width)
        else:
            return output

    @property
    @abstractmethod
    def returned_scale_shape(self):
        pass


class QuantWeightMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight_quant_config_type: Type[WeightQuantConfig],
            weight_quant_override: Module,
            weight_quant_type: Type[WeightQuantProxy],
            weight: torch.nn.Parameter,
            config_prefix: str,
            **kwargs):
        if weight_quant_override is not None:
            self.weight_quant = weight_quant_override
            self.weight_quant.add_tracked_tensor(weight)
        else:
            wqc_kwargs = filter_kwargs(config_prefix, kwargs)
            wqc = weight_quant_config_type(weight_layer=self, **wqc_kwargs)
            self.weight_quant = weight_quant_type(wqc, tracked_parameter_list_init=weight)

    @property
    @abstractmethod
    def output_channel_dim(self):
        pass

    @property
    @abstractmethod
    def out_channels(self):
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
            bias_quant_config_type: Type[BiasQuantConfig],
            bias_quant_type: Type[BiasQuantProxy],
            config_prefix: str,
            **kwargs):
        bqc_kwargs = filter_kwargs(config_prefix, kwargs)
        bqc = bias_quant_config_type(bias_layer=self, **bqc_kwargs)
        self.bias_quant = bias_quant_type(bqc)


class QuantOutputMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            output_quant_config_type: Type[ActQuantConfig],
            output_quant_override: Module,
            output_quant_type: Type[ActivationQuantProxy],
            config_prefix: str,
            **kwargs):
        if output_quant_override is not None:
             self.output_quant = output_quant_override
        else:
            oqc_kwargs = filter_kwargs(config_prefix, kwargs)
            oqc = output_quant_config_type(layer=self, **oqc_kwargs)
            self.output_quant = output_quant_type(Identity(), oqc)


class QuantWeightBiasOutputLayer(QuantOutputMixin, QuantBiasMixin, QuantWeightMixin, QuantLayer):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight,
            weight_quant_override: WeightQuantProxy,
            output_quant_override: ActivationQuantProxy,
            weight_quant_type: Type[WeightQuantProxy],
            bias_quant_type: Type[BiasQuantProxy],
            output_quant_type: Type[ActivationQuantProxy],
            weight_quant_config_type: Type[WeightQuantConfig],
            bias_quant_config_type: Type[BiasQuantConfig],
            output_quant_config_type: Type[ActQuantConfig],
            weight_quant_config_prefix: str,
            bias_quant_config_prefix: str,
            output_quant_config_prefix: str,
            return_quant_tensor: bool,
            **kwargs):
        QuantLayer.__init__(
            self,
            return_quant_tensor)
        QuantWeightMixin.__init__(
            self,
            weight_quant_config_type,
            weight_quant_override,
            weight_quant_type,
            weight,
            weight_quant_config_prefix,
            **kwargs)
        QuantBiasMixin.__init__(
            self,
            bias_quant_config_type,
            bias_quant_type,
            bias_quant_config_prefix,
            **kwargs)
        QuantOutputMixin.__init__(
            self,
            output_quant_config_type,
            output_quant_override,
            output_quant_type,
            output_quant_config_prefix,
            **kwargs)

    @abstractmethod
    def max_acc_bit_width(self, input_bit_width: Tensor, quant_weight_bit_width: Tensor):
        pass

    @abstractmethod
    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        pass

    def forward(self, x):
        output_scale = None
        output_bit_width = None
        bias_bit_width = None

        input, input_scale, input_bit_width = self.unpack_input(x)
        quant_weight, quant_weight_scale, quant_weight_bit_width = self.weight_quant(self.weight)

        if input_bit_width is not None:
            output_bit_width = self.max_acc_bit_width(input_bit_width, quant_weight_bit_width)
        if input_scale is not None:
            output_scale = input_scale * quant_weight_scale

        if self.bias is not None:
            quant_bias, _, bias_bit_width = self.bias_quant(self.bias, output_scale, output_bit_width)
            output = self.inner_forward_impl(x, quant_weight, quant_bias)
        else:
            output = self.inner_forward_impl(x, quant_weight, None)

        if bias_bit_width is not None:
            output_bit_width = torch.where(
                bias_bit_width > output_bit_width, bias_bit_width, output_bit_width)
            output_bit_width = output_bit_width + 1
        return self.pack_output(output, output_scale, output_bit_width)









