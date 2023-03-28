# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Union

from torch import Tensor
from torch.nn import MaxPool1d
from torch.nn import MaxPool2d

from brevitas.quant_tensor import QuantTensor

from .mixin.base import QuantLayerMixin


class QuantMaxPool1d(QuantLayerMixin, MaxPool1d):

    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
            return_quant_tensor: bool = True):
        MaxPool1d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode)
        QuantLayerMixin.__init__(self, return_quant_tensor=return_quant_tensor)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        if self.export_mode:
            return self.export_handler(x.value)
        x = x.set(value=super().forward(x.value))
        return self.pack_output(x)


class QuantMaxPool2d(QuantLayerMixin, MaxPool2d):

    def __init__(
            self,
            kernel_size,
            stride=None,
            padding=0,
            dilation=1,
            return_indices=False,
            ceil_mode=False,
            return_quant_tensor: bool = True):
        MaxPool2d.__init__(
            self,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode)
        QuantLayerMixin.__init__(self, return_quant_tensor=return_quant_tensor)

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
        x = x.set(value=super().forward(x.value))
        return self.pack_output(x)
