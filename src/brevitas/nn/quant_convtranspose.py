# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
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

from typing import Union, Tuple, Type, Optional

import torch
from torch import Tensor
from torch.nn import ConvTranspose1d, ConvTranspose2d
from torch.nn.functional import conv_transpose1d, conv_transpose2d

from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.quant_tensor import QuantTensor
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from .quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from .quant_layer import WeightQuantType, BiasQuantType, ActQuantType


__all__ = ['QuantConvTranspose1d', 'QuantConvTranspose2d']


class QuantConvTranspose1d(QuantWBIOL, ConvTranspose1d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            output_padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        ConvTranspose1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self._output_size = None

    @property
    def per_elem_ops(self):
        raise NotImplementedError

    @property
    def output_channel_dim(self) -> int:
        return 1

    @property
    def channelwise_separable(self) -> bool:
        raise self.groups == self.out_channels

    def forward(self, input: Union[Tensor, QuantTensor], output_size=None) -> Union[Tensor, QuantTensor]:
        self._output_size = output_size  # cache the value temporarily
        return self.forward_impl(input)

    def compute_output_padding(self, inp, output_size):
        return self._output_padding(
            inp, output_size, self.stride, self.padding, self.kernel_size)

    def conv_transpose1d_zeros_pad(
            self, x: Tensor, weight: Tensor, bias: Optional[Tensor], output_padding):
        out = conv_transpose1d(
            x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        return out

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        if self.padding_mode == 'zeros':
            output_padding = self.compute_output_padding(x, self._output_size)
            self._output_size = None  # set it back to None after consuming it
            out = self.conv_transpose1d_zeros_pad(x, quant_weight, quant_bias, output_padding)
            return out
        else:
            raise NotImplementedError(f"Padding mode {self.padding_mode} not supported.")

    def max_acc_bit_width(self, input_bit_width, weight_bit_width):
        max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_kernel_val = self.weight_quant.max_uint_value(weight_bit_width)
        group_size = self.out_channels // self.groups
        overlapping_sums = max(round(self.kernel_size[0] / self.stride[0]), 1)
        max_uint_output = max_uint_input * max_kernel_val * overlapping_sums * group_size
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width


class QuantConvTranspose2d(QuantWBIOL, ConvTranspose2d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]],
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[int, Tuple[int]] = 0,
            output_padding: Union[int, Tuple[int]] = 0,
            dilation: Union[int, Tuple[int]] = 1,
            groups: int = 1,
            bias: bool = True,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        ConvTranspose2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self._output_size = None

    @property
    def per_elem_ops(self):
        raise NotImplementedError

    @property
    def output_channel_dim(self) -> int:
        return 1

    @property
    def channelwise_separable(self) -> bool:
        raise self.groups == self.out_channels

    def forward(self, input: Union[Tensor, QuantTensor], output_size=None) -> Union[Tensor, QuantTensor]:
        self._output_size = output_size  # cache the value temporarily
        return self.forward_impl(input)

    def compute_output_padding(self, inp, output_size):
        return self._output_padding(
            inp, output_size, self.stride, self.padding, self.kernel_size)

    def conv_transpose2d_zeros_pad(
            self, x: Tensor, weight: Tensor, bias: Optional[Tensor], output_padding):
        out = conv_transpose2d(
            x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)
        return out

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        if self.padding_mode == 'zeros':
            output_padding = self.compute_output_padding(x, self._output_size)
            self._output_size = None  # set it back to None after consuming it
            out = self.conv_transpose2d_zeros_pad(x, quant_weight, quant_bias, output_padding)
            return out
        else:
            raise NotImplementedError(f"Padding mode {self.padding_mode} not supported.")

    def max_acc_bit_width(self, input_bit_width, weight_bit_width):
        max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_kernel_val = self.weight_quant.max_uint_value(weight_bit_width)
        group_size = self.out_channels // self.groups
        overlapping_sums = max(round(self.kernel_size[0] / self.stride[0]), 1)
        overlapping_sums *= max(round(self.kernel_size[1] / self.stride[1]), 1)
        max_uint_output = max_uint_input * max_kernel_val * overlapping_sums * group_size
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width
