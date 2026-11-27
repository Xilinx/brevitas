# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from brevitas.proxy.groupwise_proxy_mixin import GroupwiseProxyMixin
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.quant_tensor import GroupwiseIntQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseInt


class GroupwiseActQuantProxyFromInjector(GroupwiseProxyMixin, ActQuantProxyFromInjector):

    def __init__(self, quant_layer, quant_injector):
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOGroupwiseInt

    def create_quant_tensor(
            self,
            qt_args: Union[torch.Tensor, Tuple[Any]],
            x: Union[torch.Tensor, GroupwiseIntQuantTensor]) -> GroupwiseIntQuantTensor:
        if isinstance(qt_args, tuple):
            value, scale, zero_point, bit_width = qt_args
            out = GroupwiseIntQuantTensor(
                value,
                scale,
                zero_point,
                self.group_size,
                self.group_dim,
                bit_width,
                self.is_signed,
                self.training,
                x.shape)
        else:
            out = GroupwiseIntQuantTensor(
                qt_args,
                x.scale,
                x.zero_point,
                self.group_size,
                self.group_dim,
                x.bit_width,
                x.signed,
                self.training,
                x.shape)
        return out
