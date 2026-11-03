# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from torch import Tensor
from torch.nn import AdaptiveAvgPool2d
from torch.nn import AvgPool2d

from brevitas.inject.defaults import RoundTo8bit
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor
from brevitas.quant_tensor.int_torch_handler import adaptive_avg_pool2d_impl
from brevitas.quant_tensor.int_torch_handler import avg_pool2d_impl
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
        AvgPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override)
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

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)

        if self.export_mode:
            return self.export_handler(_unpack_quant_tensor(x))

        use_cached_input = self.cache_class is not None and (
            self.cache_inference_quant_act or self.trunc_quant.export_mode)
        if (isinstance(x, QuantTensor) or use_cached_input) and self.is_trunc_quant_enabled:
            if self.cache_inference_quant_act and isinstance(x, QuantTensor):
                self.cache_class = _CachedIO(x, self.cache_quant_io_metadata_only)
            # Pool the plain value so tracing never reconstructs QuantTensor subclasses.
            metadata = x if isinstance(x, QuantTensor) else self.cache_class
            value, scale, bit_width = avg_pool2d_impl(
                _unpack_quant_tensor(x),
                metadata.scale,
                metadata.bit_width,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override)
            y = self.trunc_quant((value, scale, metadata.zero_point, bit_width, metadata.signed))
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

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)

        # shortcut execution through the export impl during export
        if self.export_mode:
            out = self.export_handler(_unpack_quant_tensor(x))
            self._set_global_is_quant_layer(False)
            return out

        use_cached_input = self.cache_class is not None and (
            self.cache_inference_quant_act or self.trunc_quant.export_mode)
        if (isinstance(x, QuantTensor) or use_cached_input) and self.is_trunc_quant_enabled:
            if self.cache_inference_quant_act and isinstance(x, QuantTensor):
                self.cache_class = _CachedIO(x, self.cache_quant_io_metadata_only)
            # Pool the plain value so tracing never reconstructs QuantTensor subclasses.
            metadata = x if isinstance(x, QuantTensor) else self.cache_class
            value, scale, bit_width = adaptive_avg_pool2d_impl(
                _unpack_quant_tensor(x), metadata.scale, metadata.bit_width, self.output_size)
            y = self.trunc_quant((value, scale, metadata.zero_point, bit_width, metadata.signed))
        else:
            y = AdaptiveAvgPool2d.forward(self, _unpack_quant_tensor(x))

        return self.pack_output(y)
