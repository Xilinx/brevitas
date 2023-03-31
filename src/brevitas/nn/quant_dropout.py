# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Union

from torch import Tensor
from torch.nn import Dropout

from brevitas.quant_tensor import QuantTensor

from .mixin.base import QuantLayerMixin


class QuantDropout(QuantLayerMixin, Dropout):

    def __init__(self, p: float = 0.5, return_quant_tensor: bool = True):
        Dropout.__init__(self, p=p, inplace=False)
        QuantLayerMixin.__init__(self, return_quant_tensor=return_quant_tensor)

    @property
    def channelwise_separable(self) -> bool:
        return True

    @property
    def requires_export_handler(self):
        return False

    def forward(self, input: Union[Tensor, QuantTensor]):
        x = self.unpack_input(input)
        x = x.set(value=super().forward(x.value))
        return self.pack_output(x)
