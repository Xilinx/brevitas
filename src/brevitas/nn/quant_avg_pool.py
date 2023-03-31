# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import reduce
from operator import mul
from typing import Optional, Tuple, Type, Union

import torch
from torch import Tensor
from torch.nn import AdaptiveAvgPool2d
from torch.nn import AvgPool2d

from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.inject.defaults import RoundTo8bit
from brevitas.quant_tensor import QuantTensor

from .mixin.acc import AccQuantType
from .mixin.acc import TruncMixin
from .mixin.base import QuantLayerMixin


class TruncAvgPool2d(TruncMixin, QuantLayerMixin, AvgPool2d):
    """
    Quantized AvgPool2d variant that replaces the division step in the average with a right shift
    to the target bit-width through a rounding or truncation quantizer.
    Requires a QuantTensor as input and preserves the scale of the input in the output.
    """

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = None,
            trunc_quant: Optional[AccQuantType] = RoundTo8bit,
            return_quant_tensor: bool = True,
            **kwargs):
        AvgPool2d.__init__(self, kernel_size=kernel_size, stride=stride)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        TruncMixin.__init__(self, trunc_quant=trunc_quant, **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return True

    @property
    def _avg_scaling(self):
        if isinstance(self.kernel_size, tuple):
            return self.kernel_size[0] * self.kernel_size[1]
        else:
            return self.kernel_size * self.kernel_size

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            return self.export_handler(x.value)
        x = x.set(value=super(TruncAvgPool2d, self).forward(x.value))
        if self.is_trunc_quant_enabled:
            assert x.is_not_none  # check input quant tensor is filled with values
            # remove avg scaling
            rescaled_value = x.value * self._avg_scaling
            x = x.set(value=rescaled_value)
            x = x.set(bit_width=self.max_acc_bit_width(x.bit_width))
            x = self.trunc_quant(x)
        return self.pack_output(x)

    def max_acc_bit_width(self, input_bit_width):
        max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_uint_output = max_uint_input * self._avg_scaling
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width


class TruncAdaptiveAvgPool2d(TruncMixin, QuantLayerMixin, AdaptiveAvgPool2d):
    """
    Quantized AdaptiveAvgPool2d variant that replaces the division step in the average with a right shift
    to the target bit-width through a truncation quantizer.
    Requires a QuantTensor as input and preserves the scale of the input in the output.
    """

    def __init__(
            self,
            output_size: Union[int, Tuple[int, int]],
            trunc_quant: Optional[AccQuantType] = RoundTo8bit,
            return_quant_tensor: bool = True,
            cache_kernel_size_stride: bool = True,
            **kwargs):
        AdaptiveAvgPool2d.__init__(self, output_size=output_size)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        TruncMixin.__init__(self, trunc_quant=trunc_quant, **kwargs)
        self.cache_kernel_size_stride = cache_kernel_size_stride
        self._cached_kernel_size = None
        self._cached_kernel_stride = None

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return True

    @property
    def padding(self):
        return 0

    @property
    def kernel_size(self):
        return self._cached_kernel_size

    @property
    def stride(self):
        return self._cached_kernel_stride

    def compute_kernel_size_stride(self, input_shape, output_shape):
        kernel_size_list = []
        stride_list = []
        for inp, out in zip(input_shape, output_shape):
            stride = inp // out
            kernel_size = inp - (out - 1) * stride
            kernel_size_list.append(kernel_size)
            stride_list.append(stride)
        return kernel_size_list, stride_list

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y = x.set(value=super(TruncAdaptiveAvgPool2d, self).forward(x.value))
        k_size, stride = self.compute_kernel_size_stride(x.value.shape[2:], y.value.shape[2:])
        if self.cache_kernel_size_stride:
            self._cached_kernel_size = k_size
            self._cached_kernel_stride = stride
        if self.is_trunc_quant_enabled:
            assert y.is_not_none  # check input quant tensor is filled with values
            reduce_size = reduce(mul, k_size, 1)
            rescaled_value = y.value * reduce_size  # remove avg scaling
            y = y.set(value=rescaled_value)
            y = y.set(bit_width=self.max_acc_bit_width(y.bit_width, reduce_size))
            y = self.trunc_quant(y)
        return self.pack_output(y)

    def max_acc_bit_width(self, input_bit_width, reduce_size):
        max_uint_input = max_int(bit_width=input_bit_width, signed=False, narrow_range=False)
        max_uint_output = max_uint_input * reduce_size
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width
