# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjectorBase
from brevitas.proxy.groupwise_proxy_mixin import GroupwiseProxyMixin
from brevitas.quant_tensor import GroupwiseFloatQuantTensor
from brevitas.utils.quant_utils import _CachedIOGroupwiseFloat


class GroupwiseActFloatQuantProxyFromInjector(GroupwiseProxyMixin,
                                              ActFloatQuantProxyFromInjectorBase):

    def __init__(self, quant_layer, quant_injector):
        super().__init__(quant_layer, quant_injector)
        self.cache_class = _CachedIOGroupwiseFloat

    def create_quant_tensor(
            self,
            qt_args: Union[torch.Tensor, Tuple[Any]],
            x: Union[torch.Tensor, GroupwiseFloatQuantTensor]) -> GroupwiseFloatQuantTensor:
        if isinstance(qt_args, tuple):
            value, scale, zero_point, exponent_bit_width, mantissa_bit_width, exponent_bias, saturating, inf_values, nan_values = qt_args
            out = GroupwiseFloatQuantTensor(
                value,
                scale,
                zero_point,
                self.group_size,
                self.group_dim,
                exponent_bit_width,
                mantissa_bit_width,
                exponent_bias,
                saturating,
                inf_values,
                nan_values,
                self.is_signed,
                self.training,
                dequant_shape=x.shape)
        else:
            out = GroupwiseFloatQuantTensor(
                qt_args,
                x.scale,
                x.zero_point,
                self.group_size,
                self.group_dim,
                x.exponent_bit_width,
                x.mantissa_bit_width,
                x.exponent_bias,
                x.saturating,
                x.inf_values,
                x.nan_values,
                x.signed,
                self.training,
                x.shape)
        return out
