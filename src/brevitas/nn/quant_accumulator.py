# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, Type, Union

from torch.nn import Module

from brevitas.inject import BaseInjector as Injector
from brevitas.proxy.runtime_quant import AccQuantProxyProtocol
from brevitas.quant_tensor import QuantTensor

from .mixin.acc import AccQuantType
from .mixin.acc import QuantClampMixin
from .mixin.acc import TruncMixin
from .mixin.base import QuantLayerMixin


class TruncQuantAccumulator(TruncMixin, QuantLayerMixin, Module):

    def __init__(
            self,
            trunc_quant: Optional[AccQuantType] = None,
            return_quant_tensor: bool = True,
            **kwargs):
        QuantLayerMixin.__init__(self, return_quant_tensor)
        TruncMixin.__init__(self, trunc_quant=trunc_quant, **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return True

    def forward(self, input: QuantTensor):
        x = self.unpack_input(input)
        x = self.trunc_quant(x)
        return self.pack_output(x)


class ClampQuantAccumulator(QuantClampMixin, QuantLayerMixin, Module):

    def __init__(
            self,
            clamp_quant: Optional[AccQuantType] = None,
            return_quant_tensor: bool = True,
            **kwargs):
        QuantLayerMixin.__init__(self, return_quant_tensor)
        QuantClampMixin.__init__(self, clamp_quant=clamp_quant, **kwargs)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return True

    def forward(self, input: QuantTensor):
        x = self.unpack_input(input)
        x = self.clamp_quant(x)
        return self.pack_output(x)
