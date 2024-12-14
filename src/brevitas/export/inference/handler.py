# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch

from brevitas import is_dynamo_compiling
from brevitas.function.ops import max_float
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjectorBase
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_runtime_quant import GroupwiseActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.quant.experimental.mx_quant_ocp import GroupwiseActQuantProxyFromInjector
from brevitas.utils.torch_utils import float_internal_scale


class InferenceHandler(torch.nn.Module, ABC):

    def attach_debug_info(self, module):
        pass

    @abstractmethod
    def prepare_for_export(self, module):
        pass

    @abstractmethod
    def quantize(self, x):
        pass

    @abstractmethod
    def dequantize(self, x):
        pass


class IntInferencetHandler(InferenceHandler):
    handled_layer = (ActQuantProxyFromInjector, BiasQuantProxyFromInjector)

    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.ones(0))

    def attach_debug_info(self, module):
        pass

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.scale = module.scale()
            self.zero_point = module.zero_point().to(self.scale.device)
            self.bit_width = module.bit_width()
            self.min_clamp = min_int(module.is_signed, module.is_narrow_range, self.bit_width)
            self.max_clamp = max_int(module.is_signed, module.is_narrow_range, self.bit_width)

    def quantize(self, x, scale, zero_point):
        return torch.clamp(torch.round(x / scale + zero_point), self.min_clamp, self.max_clamp)

    def dequantize(self, x, scale, zero_point):
        return (x - zero_point) * scale

    def forward(self, x, unused_scale=None) -> Tuple[torch.Tensor]:
        return self.dequantize(self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point), self.scale, self.zero_point, self.bit_width


class IntWeightInferencetHandler(IntInferencetHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.register_buffer('cached_weight', torch.ones(1))

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.cached_weight = None
            super().prepare_for_export(module)
            if module._cached_weight is not None and not module.cache_inference_quant_weight_metadata_only:
                self.cached_weight = module._cached_weight.value

    def forward(self, x) -> Tuple[torch.Tensor]:
        if self.cached_weight is not None:
            x = self.cached_weight
        else:
            x = self.dequantize(
                self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point)
        return x, self.scale, self.zero_point, self.bit_width


class GroupwiseIntInferenceHandler(IntInferencetHandler):
    handled_layer = GroupwiseActQuantProxyFromInjector

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy
            self.group_dim = module.group_dim

    def forward(self, x, unused_scale=None) -> Tuple[torch.Tensor]:
        x, *other = self.module_forward(x)
        if is_dynamo_compiling:
            start_dim = self.group_dim if self.group_dim != -1 else -2
            x = x.flatten(start_dim, start_dim + 1)
        output_args = tuple([x] + list(other))
        return output_args


class GroupwiseIntWeightInferenceHandler(IntWeightInferencetHandler):
    handled_layer = GroupwiseWeightQuantProxyFromInjector

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        self.input_view = module.input_view_impl
        self.flattened_view = module.apply_input_view

    def forward(self, x, unused_scale=None) -> Tuple[torch.Tensor]:
        x = self.input_view(x)
        if self.scale.shape != ():
            scale = self.input_view(self.scale)
        else:
            scale = self.scale
        if self.zero_point.shape != ():
            zero_point = self.input_view(self.zero_point)
        else:
            zero_point = self.zero_point
        out = self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)
        if is_dynamo_compiling:
            out = self.flattened_view(out)
        return out, scale, zero_point, self.bit_width


class FloatInferencetHandler(InferenceHandler):
    handled_layer = (ActFloatQuantProxyFromInjector, BiasQuantProxyFromInjector)

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.scale = module.scale()
            self.zero_point = module.zero_point().to(self.scale.device)
            self.exponent_bit_width = module.exponent_bit_width()
            self.mantissa_bit_width = module.mantissa_bit_width()
            self.exponent_bias = module.exponent_bias()
            self.saturating = module.is_saturating()
            self.inf_values = module.inf_values()
            self.nan_values = module.nan_values()
            self.eps = torch.finfo(self.scale.dtype).tiny
            if hasattr(module.tensor_quant, 'float_to_int_impl'):
                self.float_to_int_impl = module.tensor_quant.float_to_int_impl
                self.float_clamp_impl = module.tensor_quant.float_clamp_impl
            elif hasattr(module, 'fused_activation_quant_proxy'):
                self.float_to_int_impl = module.fused_activation_quant_proxy.tensor_quant.float_to_int_impl
                self.float_clamp_impl = module.fused_activation_quant_proxy.tensor_quant.float_clamp_impl

            self.max_clamp = max_float(
                self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias)
            self.min_clamp = -self.max_clamp
            self.fp_internal_scale_min = 1. - self.exponent_bias - self.mantissa_bit_width
            self.max_value = max_float(
                self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias)
            self.min_value = torch.tensor(0.) if not module.is_signed else -self.max_value

    def quantize(self, x, scale, zero_point):
        # Compute masks
        inf_mask = x.isinf()
        p_max_val_mask = x > self.max_value
        n_max_val_mask = -x > self.max_value
        # Quantize
        x = x / scale
        internal_scale = float_internal_scale(
            x, self.mantissa_bit_width, self.fp_internal_scale_min, self.eps)
        x = internal_scale * self.float_to_int_impl(x / internal_scale)

        # Clamp
        x = self.float_clamp_impl.saturating_clamp(x, self.max_value, self.min_value)
        if not self.saturating:
            x = self.float_clamp_impl.inf_nan_clamp(x, inf_mask, p_max_val_mask, n_max_val_mask)

        return x

    def dequantize(self, x, scale, zero_point):
        return (x - zero_point) * scale

    def forward(self, x) -> Tuple[torch.Tensor]:
        return self.dequantize(self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point), self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class FloatWeightInferencetHandler(FloatInferencetHandler):
    handled_layer = WeightFloatQuantProxyFromInjector

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.cached_weight = None
            super().prepare_for_export(module)
            if module._cached_weight is not None and not module.cache_inference_quant_weight_metadata_only:
                self.cached_weight = module._cached_weight.value

    def forward(self, x) -> Tuple[torch.Tensor]:
        if self.cached_weight is not None:
            x = self.cached_weight
        else:
            x = self.dequantize(
                self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point)
        return x, self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class GroupwiseFloatInferenceHandler(FloatInferencetHandler):
    handled_layer = GroupwiseActFloatQuantProxyFromInjector

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy
            self.group_dim = module.group_dim

    def forward(self, x, unused_scale=None) -> Tuple[torch.Tensor]:
        x, *other = self.module_forward(x)
        if is_dynamo_compiling:
            start_dim = self.group_dim if self.group_dim != -1 else -2
            x = x.flatten(start_dim, start_dim + 1)
        output_args = tuple([x] + list(other))
        return output_args


class GroupwiseFloatWeightInferenceHandler(FloatWeightInferencetHandler):
    handled_layer = GroupwiseWeightFloatQuantProxyFromInjector

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        self.input_view = module.input_view_impl
        self.flattened_view = module.apply_input_view

    def forward(self, x, unused_scale=None) -> Tuple[torch.Tensor]:
        x = self.input_view(x)
        if self.scale.shape != ():
            scale = self.input_view(self.scale)
        else:
            scale = self.scale
        if self.zero_point.shape != ():
            zero_point = self.input_view(self.zero_point)
        else:
            zero_point = self.zero_point
        out = self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)
        if is_dynamo_compiling:
            out = self.flattened_view(out)
        return out, scale, zero_point, self.bit_width
