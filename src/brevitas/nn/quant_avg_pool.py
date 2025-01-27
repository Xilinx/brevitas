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
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.quant_utils import _CachedIO

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
            ceil_mode: bool = False,
            count_include_pad: bool = True,
            divisor_override: Optional[int] = None,
            trunc_quant: Optional[AccQuantType] = RoundTo8bit,
            return_quant_tensor: bool = True,
            **kwargs):
        AvgPool2d.__init__(self, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        TruncMixin.__init__(self, trunc_quant=trunc_quant, **kwargs)
        self.cache_inference_quant_act = False
        self.cache_quant_io_metadata_only = True
        self.cache_class = None

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

    # TODO: Replace with functional call
    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)

        if self.export_mode:
            return self.export_handler(_unpack_quant_tensor(x))

        if (isinstance(x, QuantTensor) or
                self.cache_class is not None) and self.is_trunc_quant_enabled:
            if self.cache_inference_quant_act:
                self.cache_class = _CachedIO(x, self.cache_quant_io_metadata_only)
            if not isinstance(x, QuantTensor):
                x = self.cache_class.quant_tensor.set(value=x)
            y = AvgPool2d.forward(self, x)
            y = self.trunc_quant(y)
        else:
            y = AvgPool2d.forward(self, _unpack_quant_tensor(x))

        return self.pack_output(y)


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
            **kwargs):
        AdaptiveAvgPool2d.__init__(self, output_size=output_size)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        TruncMixin.__init__(self, trunc_quant=trunc_quant, **kwargs)
        self.cache_inference_quant_act = False
        self.cache_quant_io_metadata_only = True
        self.cache_class = None

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return True

    @property
    def padding(self):
        return 0

    @staticmethod
    def compute_kernel_size_stride(input_shape, output_shape):
        kernel_size_list = []
        stride_list = []
        for inp, out in zip(input_shape, output_shape):
            stride = inp // out
            kernel_size = inp - (out - 1) * stride
            kernel_size_list.append(kernel_size)
            stride_list.append(stride)
        return kernel_size_list, stride_list

    # TODO: Replace with functional call
    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)

        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler(_unpack_quant_tensor(x))
            self._set_global_is_quant_layer(False)
            return out

        if (isinstance(x, QuantTensor) or
                self.cache_class is not None) and self.is_trunc_quant_enabled:
            if self.cache_inference_quant_act:
                self.cache_class = _CachedIO(x, self.cache_quant_io_metadata_only)
            if not isinstance(x, QuantTensor):
                x = self.cache_class.quant_tensor.set(value=x)
            y = AdaptiveAvgPool2d.forward(self, x)
            y = self.trunc_quant(y)
        else:
            y = AdaptiveAvgPool2d.forward(self, _unpack_quant_tensor(x))

        return self.pack_output(y)
