# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Union

from torch import Tensor
from torch.nn import Upsample
from torch.nn import UpsamplingBilinear2d
from torch.nn import UpsamplingNearest2d
from torch.nn.functional import interpolate

from brevitas.function import round_ste
from brevitas.quant_tensor import QuantTensor

from .mixin.base import QuantLayerMixin


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

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        if self.mode != 'nearest':
            # round interpolated values to scale
            assert x.scale is not None, 'Input scale factor required to interpolate correctly'
            y_value = round_ste(y_value / x.scale) * x.scale
        y = x.set(value=y_value)
        return self.pack_output(y)


class QuantUpsamplingBilinear2d(QuantLayerMixin, UpsamplingBilinear2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool = True):
        UpsamplingBilinear2d.__init__(self, size=size, scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        # round interpolated values to scale
        assert x.scale is not None, 'Input scale factor required to interpolate correctly'
        y_value = round_ste(y_value / x.scale) * x.scale
        y = x.set(value=y_value)
        return self.pack_output(y)


class QuantUpsamplingNearest2d(QuantLayerMixin, UpsamplingNearest2d):

    def __init__(self, size=None, scale_factor=None, return_quant_tensor: bool = True):
        UpsamplingNearest2d.__init__(self, size=size, scale_factor=scale_factor)
        QuantLayerMixin.__init__(self, return_quant_tensor)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            out = self.export_handler(x.value)
            self._set_global_is_quant_layer(False)
            return out
        y_value = interpolate(x.value, self.size, self.scale_factor, self.mode, self.align_corners)
        y = x.set(value=y_value)
        return self.pack_output(y)
