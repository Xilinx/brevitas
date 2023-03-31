# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional, Tuple, Type, Union

import torch
from torch import Tensor
from torch.nn import Conv1d
from torch.nn import Conv2d
from torch.nn import functional as F
from torch.nn.functional import conv2d

from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from brevitas.quant_tensor import QuantTensor

from .quant_layer import ActQuantType
from .quant_layer import BiasQuantType
from .quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from .quant_layer import WeightQuantType

__all__ = ['QuantConv1d', 'QuantConv2d']


class QuantConv1d(QuantWBIOL, Conv1d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_type: str = 'standard',
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        Conv1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
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
        assert self.padding_mode == 'zeros'
        assert not (padding_type == 'same' and padding != 0)
        self.padding_type = padding_type

    @property
    def per_elem_ops(self):
        return 2 * self.kernel_size[0] * (self.in_channels // self.groups)

    @property
    def output_channel_dim(self):
        if self.transposed:
            return 1
        else:
            return 0

    @property
    def channelwise_separable(self) -> bool:
        return self.groups == self.in_channels

    def conv1d_zeros_pad(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]):
        out = F.conv1d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def conv1d_same_zeros_pad(self, x, weight, bias):
        ih = x.size()[-1]
        kh = weight.size()[-1]
        sh = self.stride[0]
        oh = math.ceil(ih / sh)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        if pad_h > 0:
            x = F.pad(x, [pad_h // 2, pad_h - pad_h // 2])
        out = F.conv1d(x, weight, bias, self.stride, 0, self.dilation, self.groups)
        return out

    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        return self.forward_impl(input)

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        if self.padding_type == 'standard':
            return self.conv1d_zeros_pad(x, quant_weight, quant_bias)
        elif self.padding_type == 'same':
            return self.conv1d_same_zeros_pad(x, quant_weight, quant_bias)
        else:
            raise NotImplementedError(f"Padding type {self.padding_type} not supported.")

    def max_acc_bit_width(self, input_bit_width, weight_bit_width):
        max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_kernel_val = self.weight_quant.max_uint_value(weight_bit_width)
        group_size = self.out_channels // self.groups
        max_uint_output = max_uint_input * max_kernel_val * self.kernel_size[0] * group_size
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width


class QuantConv2d(QuantWBIOL, Conv2d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            padding_type: str = 'standard',
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        Conv2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
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
        assert self.padding_mode == 'zeros'
        assert not (padding_type == 'same' and padding != 0)
        self.padding_type = padding_type

    @property
    def per_elem_ops(self):
        flat_kernel_size = self.kernel_size[0] * self.kernel_size[1]
        return 2 * flat_kernel_size * (self.in_channels // self.groups)

    @property
    def output_channel_dim(self):
        if self.transposed:
            raise RuntimeError("Transposed kernels not supported")
        return 0

    @property
    def channelwise_separable(self) -> bool:
        return self.groups == self.in_channels

    def conv2d_zeros_pad(self, x: Tensor, weight: Tensor, bias: Tensor):
        out = conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def conv2d_same_zeros_pad(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]):
        ih, iw = x.size()[-2:]
        kh, kw = weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        out = F.conv2d(x, weight, bias, self.stride, 0, self.dilation, self.groups)
        return out

    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        return self.forward_impl(input)

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        if self.padding_type == 'standard':
            return self.conv2d_zeros_pad(x, quant_weight, quant_bias)
        elif self.padding_type == 'same':
            return self.conv2d_same_zeros_pad(x, quant_weight, quant_bias)
        else:
            raise RuntimeError(f"Padding type {self.padding_type} not supported.")

    def max_acc_bit_width(self, input_bit_width: Tensor, weight_bit_width: Tensor):
        max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_kernel_val = self.weight_quant.max_uint_value(weight_bit_width)
        group_size = self.out_channels // self.groups
        kernel_size = self.kernel_size[0] * self.kernel_size[1]
        max_uint_output = max_uint_input * max_kernel_val * kernel_size * group_size
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width
