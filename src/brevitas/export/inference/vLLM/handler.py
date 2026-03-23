# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.function_wrapper.shape import dynamic_over_sub_channel_block_view
from brevitas.core.function_wrapper.shape import OverOutputFeaturesView
from brevitas.core.function_wrapper.shape import PermuteDims
from brevitas.core.restrict_val import RestrictValueType
from brevitas.function.shape import over_output_features
from brevitas.proxy.float_runtime_quant import DynamicActFloatQuantProxyFromInjector
from brevitas.utils.quant_utils import groupwise_dequant_expand

from ..handler import DynamicFloatInferenceHandler
from ..handler import DynamicScaleZeroPointMixin
from ..handler import GroupwiseFloatInferenceHandler
from ..handler import GroupwiseIntInferenceHandler

EPS = 1e-16


def maybe_permute(x: Tensor, permute_dims: Optional[Tuple[int, ...]]) -> Tensor:
    if permute_dims is None:
        return x
    else:
        return x.permute(*permute_dims).contiguous()


class StandaloneGroupwiseQuantMixin(DynamicScaleZeroPointMixin):

    def compute_scale(self, x: Tensor, group_dim: Optional[int] = None) -> Tensor:
        if group_dim is not None:
            max_abs = torch.max(torch.abs(x), dim=group_dim, keepdim=True)[0]
        else:
            max_abs = torch.max(torch.abs(x))
        scale = torch.clamp(max_abs, EPS)
        threshold = self.threshold
        if self.scaling_restriction == RestrictValueType.POWER_OF_TWO:
            scale = torch.pow(2, torch.floor(torch.log2(scale)))
        if self.threshold_restriction == RestrictValueType.POWER_OF_TWO:
            threshold = torch.clamp(torch.pow(2, torch.floor(torch.log2(threshold))), EPS)
        scale = scale / threshold
        return scale


class vLLMGroupwiseIntInferenceHandler(GroupwiseIntInferenceHandler, StandaloneGroupwiseQuantMixin):

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        inp_shape = x.shape
        x = dynamic_over_sub_channel_block_view(x, self.group_size, self.group_dim)
        group_dim = self.group_dim + 1 if self.group_dim > 0 else self.group_dim
        scale = self.compute_scale(x, group_dim)
        zero_point = torch.zeros(()).type_as(x)
        out = self.inner_forward(x, scale, zero_point)
        out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim, inp_shape)[0]
        return out, scale, zero_point, self.bit_width


class vLLMGroupwiseFloatInferenceHandler(GroupwiseFloatInferenceHandler,
                                         StandaloneGroupwiseQuantMixin):

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        inp_shape = x.shape
        x = dynamic_over_sub_channel_block_view(x, self.group_size, self.group_dim)
        group_dim = self.group_dim + 1 if self.group_dim > 0 else self.group_dim
        scale = self.compute_scale(x, group_dim)
        zero_point = torch.zeros((), dtype=x.dtype, device=x.device)
        out = self.inner_forward(x, scale, zero_point)
        out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim, inp_shape)[0]
        return out, scale, zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class vLLMDynamicPerRowFloatInferenceHandler(DynamicFloatInferenceHandler,
                                             StandaloneGroupwiseQuantMixin):
    handled_layer = DynamicActFloatQuantProxyFromInjector

    def __init__(self) -> None:
        DynamicFloatInferenceHandler.__init__(self)
        StandaloneGroupwiseQuantMixin.__init__(self)
        self.register_buffer("permute_dims", None)
        self.stats_reduce_dim = 1

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        for name, submodule in module.named_modules():
            if name.endswith('scaling_stats_input_view_shape_impl'):
                assert type(submodule) == OverOutputFeaturesView, "Only per-row dynamic quantization is supported"
                if hasattr(submodule, 'permute_dims'):
                    self.permute_dims = submodule.permute_dims
                else:
                    self.permute_dims = None

    def dynamic_broadcast(self, x: Tensor, shape: List[int]) -> Tensor:
        return x.view(*shape[:-1], 1)

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        out = self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)
        return out

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        x = maybe_permute(x, self.permute_dims)
        x_shape = over_output_features(x)
        scale = self.compute_scale(x.reshape(x_shape), self.stats_reduce_dim)
        scale = self.dynamic_broadcast(scale, x.shape)
        zero_point = torch.zeros((), dtype=x.dtype, device=x.device)
        out = self.inner_forward(x, scale, zero_point)
        return out, scale, zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values
