# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch

from brevitas.function.ops import max_float
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjectorBase
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
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

    def attach_debug_info(self, module):
        pass

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.scale = module.scale()
            self.zero_point = module.zero_point().to(self.scale.device)
            self.bit_width = module.bit_width()
            self.min_clamp = min_int(module.is_signed, module.is_narrow_range, self.bit_width)
            self.max_clamp = max_int(module.is_signed, module.is_narrow_range, self.bit_width)

    def quantize(self, x):
        return torch.clamp(
            torch.round(x / self.scale + self.zero_point), self.min_clamp, self.max_clamp)

    def dequantize(self, x):
        return (x - self.zero_point) * self.scale

    def forward(self, x, unused_scale=None) -> Tuple[torch.Tensor]:
        return self.dequantize(self.quantize(x)), self.scale, self.zero_point, self.bit_width


class IntWeightInferencetHandler(IntInferencetHandler):
    handled_layer = WeightQuantProxyFromInjector

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
            x = self.dequantize(self.quantize(x))
        return x, self.scale, self.zero_point, self.bit_width


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

    def quantize(self, x):
        # Compute masks
        inf_mask = x.isinf()
        p_max_val_mask = x > self.max_value
        n_max_val_mask = -x > self.max_value

        # Quantize
        x = x / self.scale
        internal_scale = float_internal_scale(
            x, self.mantissa_bit_width, self.fp_internal_scale_min, self.eps)
        x = internal_scale * self.float_to_int_impl(x / internal_scale)

        # Clamp
        x = self.float_clamp_impl.saturating_clamp(x, self.max_value, self.min_value)
        if not self.saturating:
            x = self.float_clamp_impl.inf_nan_clamp(x, inf_mask, p_max_val_mask, n_max_val_mask)

        return x

    def dequantize(self, x):
        return (x - self.zero_point) * self.scale

    def forward(self, x) -> Tuple[torch.Tensor]:
        return self.dequantize(self.quantize(x)), self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


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
            x = self.dequantize(self.quantize(x))
        return x, self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values
