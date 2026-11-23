# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional
from typing import Union

import torch
from torch import Tensor
from torch.nn import Upsample
from torch.nn import UpsamplingBilinear2d
from torch.nn import UpsamplingNearest2d
from torch.nn.functional import interpolate

from brevitas import is_dynamo_compiling
from brevitas.function import round_ste
from brevitas.quant_tensor import _unpack_quant_tensor
from brevitas.quant_tensor import QuantTensor

from .mixin.base import QuantLayerMixin


def _round_to_input_scale(y_value: Tensor, scale: Optional[Tensor]) -> Tensor:
    """
    Round interpolated values back onto the input quantization grid.

    With no cached scale (e.g. a plain-Tensor eager input, which has no grid),
    the values are returned unchanged. Under Dynamo, plain ``torch.round`` is
    used: the straight-through estimator is irrelevant for the no-grad export
    pass, and tracing the custom ``round_ste`` autograd function is not needed.
    """
    if scale is None:
        return y_value
    if is_dynamo_compiling():
        return torch.round(y_value / scale) * scale
    return round_ste(y_value / scale) * scale


class QuantUpsample(QuantLayerMixin, Upsample):

    def __init__(
            self,
            size=None,
            scale_factor=None,
            mode='nearest',
            align_corners=None,
            return_quant_tensor: bool = True):
        Upsample.__init__(
            self, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        # Set by the export managers during the eager pass that precedes
        # tracing (see _cache_inp_out), so the input scale is available when
        # tracing hands this layer a plain tensor instead of a QuantTensor.
        self.cache_inference_quant_act = False
        self._cached_input_scale = None

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(_unpack_quant_tensor(x))
            self._set_global_is_quant_layer(False)
            return out
        if self.cache_inference_quant_act and isinstance(x, QuantTensor):
            # Only the scale is needed to round bilinear outputs back onto the
            # input grid; caching it directly also covers FloatQuantTensor
            # inputs, which the IntQuantTensor-only _CachedIO does not.
            self._cached_input_scale = x.scale
        y_value = interpolate(
            _unpack_quant_tensor(x), self.size, self.scale_factor, self.mode, self.align_corners)
        if isinstance(x, QuantTensor):
            if self.mode != 'nearest':
                # round interpolated values to scale
                y_value = _round_to_input_scale(y_value, x.scale)
            y = x.set(value=y_value)
            return self.pack_output(y)
        # Plain Tensor input: under Dynamo, QuantTensor creation is disabled
        # so upstream layers hand us a FakeTensor, and in plain eager mode
        # there is no QuantTensor to propagate. Interpolate the plain value,
        # rounding bilinear outputs onto the cached input grid when available.
        if self.mode != 'nearest':
            y_value = _round_to_input_scale(y_value, self._cached_input_scale)
        return y_value


class QuantUpsamplingBilinear2d(QuantLayerMixin, UpsamplingBilinear2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool = True):
        UpsamplingBilinear2d.__init__(self, size=size, scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        # See QuantUpsample.
        self.cache_inference_quant_act = False
        self._cached_input_scale = None

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(_unpack_quant_tensor(x))
            self._set_global_is_quant_layer(False)
            return out
        if self.cache_inference_quant_act and isinstance(x, QuantTensor):
            self._cached_input_scale = x.scale
        y_value = interpolate(
            _unpack_quant_tensor(x), self.size, self.scale_factor, self.mode, self.align_corners)
        if isinstance(x, QuantTensor):
            # round interpolated values to scale
            y_value = _round_to_input_scale(y_value, x.scale)
            y = x.set(value=y_value)
            return self.pack_output(y)
        # Plain Tensor input: see QuantUpsample.
        y_value = _round_to_input_scale(y_value, self._cached_input_scale)
        return y_value


class QuantUpsamplingNearest2d(QuantLayerMixin, UpsamplingNearest2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool = True):
        UpsamplingNearest2d.__init__(self, size=size, scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)
        # See QuantUpsample.
        self.cache_inference_quant_act = False
        self._cached_input_scale = None

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(_unpack_quant_tensor(x))
            self._set_global_is_quant_layer(False)
            return out
        if self.cache_inference_quant_act and isinstance(x, QuantTensor):
            self._cached_input_scale = x.scale
        y_value = interpolate(
            _unpack_quant_tensor(x), self.size, self.scale_factor, self.mode, self.align_corners)
        if isinstance(x, QuantTensor):
            # Nearest-neighbour upsampling does not change the quantization
            # grid, so no rounding step is needed.
            y = x.set(value=y_value)
            return self.pack_output(y)
        # Plain Tensor input: see QuantUpsample.
        return y_value
