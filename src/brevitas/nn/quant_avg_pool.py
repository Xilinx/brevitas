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

from typing import Optional, Union, Type, Tuple
from operator import mul
from functools import reduce

import torch
from torch import Tensor
from torch.nn import AvgPool2d, AdaptiveAvgPool2d

from brevitas.function.ops_ste import ceil_ste
from brevitas.function.ops import max_int
from brevitas.quant_tensor import QuantTensor
from brevitas.inject.defaults import TruncTo8bit
from .mixin.base import QuantLayerMixin
from .mixin.acc import QuantTruncMixin, AccQuantType


class QuantAvgPool2d(QuantTruncMixin, QuantLayerMixin, AvgPool2d):

    def __init__(
            self,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = None,
            trunc_quant: Optional[AccQuantType] = TruncTo8bit,
            return_quant_tensor: bool = True,
            **kwargs):
        AvgPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantTruncMixin.__init__(
            self,
            trunc_quant=trunc_quant,
            **kwargs)

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
        x = x.set(value=super(QuantAvgPool2d, self).forward(x.value))
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


class QuantAdaptiveAvgPool2d(QuantTruncMixin, QuantLayerMixin, AdaptiveAvgPool2d):

    def __init__(
            self,
            output_size: Union[int, Tuple[int, int]],
            trunc_quant: Optional[AccQuantType] = TruncTo8bit,
            return_quant_tensor: bool = True,
            cache_kernel_size_stride: bool = True,
            **kwargs):
        AdaptiveAvgPool2d.__init__(self, output_size=output_size)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantTruncMixin.__init__(
            self,
            trunc_quant=trunc_quant,
            **kwargs)
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
        y = x.set(value=super(QuantAdaptiveAvgPool2d, self).forward(x.value))
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
