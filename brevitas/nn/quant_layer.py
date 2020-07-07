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
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import Identity

from brevitas.quant_tensor import QuantTensor
from brevitas.core.quant import IdentityQuant
from brevitas.proxy import WeightQuantProxy, BiasQuantProxy, ActivationQuantProxy
from brevitas.proxy.config import ActQuantConfig, WeightQuantConfig, BiasQuantConfig
from brevitas.proxy.spec import OutputQuantSpec, WeightQuantSpec, BiasQuantSpec
from brevitas.proxy.spec import OutputQuantConfigSpec, WeightQuantConfigSpec, BiasQuantConfigSpec

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
            weight_quant,
            weight: torch.nn.Parameter,
            **kwargs):
        if isinstance(weight_quant, WeightQuantProxy):
            self.weight_quant = weight_quant
            self.weight_quant.add_tracked_tensor(weight)
        elif isinstance(weight_quant, WeightQuantSpec):
            if isinstance(weight_quant.config, WeightQuantConfig):
                wqc = weight_quant.config
            elif isinstance(weight_quant.config, WeightQuantConfigSpec):
                wqc_kwargs = filter_kwargs(weight_quant.config.prefix, kwargs)
                wqc = weight_quant.config.type(weight_layer=self, **wqc_kwargs)
            else:
                raise RuntimeError
            self.weight_quant = weight_quant.type(wqc, tracked_parameter_list_init=weight)
        else:
            raise RuntimeError

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
            bias_quant,
            **kwargs):
        if isinstance(bias_quant, BiasQuantProxy):
            self.bias_quant = bias_quant
        elif isinstance(bias_quant, BiasQuantSpec):
            if isinstance(bias_quant.config, BiasQuantConfig):
                bqc = bias_quant.config
            elif isinstance(bias_quant.config, BiasQuantConfigSpec):
                bqc_kwargs = filter_kwargs(bias_quant.config.prefix, kwargs)
                bqc = bias_quant.config.type(bias_layer=self, **bqc_kwargs)
            else:
                raise RuntimeError
            self.bias_quant = bias_quant.type(bqc)
        else:
            raise RuntimeError


class QuantOutputMixin(object):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            output_quant: Union[ActivationQuantProxy, OutputQuantSpec],
            **kwargs):
        if isinstance(output_quant, ActivationQuantProxy):
            self.output_quant = output_quant
        elif isinstance(output_quant, OutputQuantSpec):
            if isinstance(output_quant.config, ActQuantConfig):
                oqc = output_quant.config
            elif isinstance(output_quant.config, OutputQuantConfigSpec):
                oqc_kwargs = filter_kwargs(output_quant.config.prefix, kwargs)
                oqc = output_quant.config.type(layer=self, **oqc_kwargs)
            else:
                raise RuntimeError
            self.output_quant = output_quant.type(Identity(), oqc)
        else:
            raise RuntimeError


class QuantWeightBiasOutputLayer(QuantOutputMixin, QuantBiasMixin, QuantWeightMixin, QuantLayer):
    __metaclass__ = ABCMeta

    def __init__(
            self,
            weight,
            weight_quant: Union[WeightQuantProxy, WeightQuantSpec],
            bias_quant: Union[BiasQuantProxy, BiasQuantSpec],
            output_quant: Union[ActivationQuantProxy, OutputQuantSpec],
            return_quant_tensor: bool,
            **kwargs):
        QuantLayer.__init__(self, return_quant_tensor)
        QuantWeightMixin.__init__(self, weight_quant, weight, **kwargs)
        QuantBiasMixin.__init__(self, bias_quant, **kwargs)
        QuantOutputMixin.__init__(self, output_quant, **kwargs)

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

        output, output_scale, output_bit_width = self.output_quant(output, output_scale, output_bit_width)
        return self.pack_output(output, output_scale, output_bit_width)









