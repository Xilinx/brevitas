# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn

from brevitas.function import compute_max_mantissa
from brevitas.function.ops import max_float
from brevitas.function.ops import max_int
from brevitas.function.ops import min_int
from brevitas.proxy.float_parameter_quant import WeightFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import ActFloatQuantProxyFromInjector
from brevitas.proxy.float_runtime_quant import DynamicActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_runtime_quant import GroupwiseActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector
from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import DynamicActQuantProxyFromInjector
from brevitas.quant.experimental.mx_quant_ocp import GroupwiseActQuantProxyFromInjector
from brevitas.utils.quant_utils import groupwise_dequant_expand
from brevitas.utils.torch_utils import float_internal_scale


class InferenceHandler(torch.nn.Module, ABC):

    def attach_debug_info(self, module: nn.Module):
        pass

    @abstractmethod
    def prepare_for_export(self, module: nn.Module):
        pass

    @abstractmethod
    def quantize(self, x: Tensor):
        pass

    @abstractmethod
    def dequantize(self, x: Tensor):
        pass


class IntInferencetHandler(InferenceHandler):
    handled_layer = (ActQuantProxyFromInjector, BiasQuantProxyFromInjector)

    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.ones(0))

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            scale = module.scale_() if hasattr(module, 'scale_') else module.scale()
            zero_point = module.zero_point_() if hasattr(module,
                                                         'zero_point_') else module.zero_point()
            # Continguous is used to be extra-safe with torch.compile
            self.scale = scale.contiguous()
            self.zero_point = zero_point.contiguous()

            self.zero_point = self.zero_point.to(self.scale.device)
            self.bit_width = module.bit_width()
            self.min_clamp = min_int(module.is_signed, module.is_narrow_range, self.bit_width)
            self.max_clamp = max_int(module.is_signed, module.is_narrow_range, self.bit_width)
            if hasattr(module.tensor_quant, 'int_quant'):
                self.float_to_int_impl = module.tensor_quant.int_quant.float_to_int_impl
            elif hasattr(module, 'fused_activation_quant_proxy'):
                self.float_to_int_impl = module.fused_activation_quant_proxy.tensor_quant.int_quant.float_to_int_impl

    def quantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        return torch.clamp(
            self.float_to_int_impl(x / scale + zero_point), self.min_clamp, self.max_clamp)

    def dequantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return (x - zero_point) * scale

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        return self.dequantize(self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point), self.scale, self.zero_point, self.bit_width


class IntWeightInferencetHandler(IntInferencetHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.register_buffer('cached_weight', torch.ones(1))

    def prepare_for_export(self, module: nn.Module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            if module._cached_weight is not None and not module.cache_inference_quant_weight_metadata_only:
                self.cached_weight = module._cached_weight.value
            else:
                self.cached_weight = None

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        if self.cached_weight is not None:
            x = self.cached_weight
        else:
            x = self.inner_forward(x, self.scale, self.zero_point)

        return x, self.scale, self.zero_point, self.bit_width


class DynamicIntInferenceHandler(IntInferencetHandler):
    handled_layer = DynamicActQuantProxyFromInjector

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        return self.module_forward(x)


class GroupwiseIntInferenceHandler(IntInferencetHandler):
    handled_layer = GroupwiseActQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant
            self.group_dim = module.group_dim

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        inp_shape = x.shape
        x, scale, zero_point, *other = self.module_forward(x)

        # If we skip quant tensor, we return the flattened version of the groupwise tensor
        if self.skip_create_quant_tensor:
            x = groupwise_dequant_expand(x, scale, zero_point, self.group_dim, inp_shape)[0]
        output_args = tuple([x, scale, zero_point] + list(other))
        return output_args


class GroupwiseIntWeightInferenceHandler(IntWeightInferencetHandler):
    handled_layer = GroupwiseWeightQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            self.group_dim = module.group_dim
            self.input_view = module.input_view_impl

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        scale = self.scale
        if scale.shape != ():
            scale = self.input_view(scale)
        zero_point = self.zero_point
        if zero_point.shape != ():
            zero_point = self.input_view(zero_point)

        if self.cached_weight is not None:
            out = self.cached_weight
        else:
            inp_shape = x.shape
            x = self.input_view(x)
            out = self.inner_forward(x, scale, zero_point)

            # If we skip quant tensor, we return the flattened version of the groupwise tensor
            out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim, inp_shape)[0]
        return out, scale, zero_point, self.bit_width


class FloatInferencetHandler(InferenceHandler):
    handled_layer = (ActFloatQuantProxyFromInjector, BiasQuantProxyFromInjector)

    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.ones(0))

    def prepare_for_export(self, module):
        if module.is_quant_enabled:
            self.scale = module.scale_() if hasattr(module, 'scale_') else module.scale()
            self.zero_point = module.zero_point_() if hasattr(
                module, 'zero_point_') else module.zero_point()
            # Continguous is used to be extra-safe with torch.compile
            self.zero_point = self.zero_point.contiguous()
            self.scale = self.scale.contiguous()
            self.zero_point = self.zero_point.to(self.scale.device)
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
                self.max_available_float = module.tensor_quant.float_clamp_impl.max_available_float
            elif hasattr(module, 'fused_activation_quant_proxy'):
                self.float_to_int_impl = module.fused_activation_quant_proxy.tensor_quant.float_to_int_impl
                self.float_clamp_impl = module.fused_activation_quant_proxy.tensor_quant.float_clamp_impl
                self.max_available_float = module.fused_activation_quant_proxy.tensor_quant.float_clamp_impl.max_available_float

            self.pre_compute_max_mantissa = compute_max_mantissa(self.mantissa_bit_width)
            self.max_clamp = max_float(
                self.exponent_bit_width, self.pre_compute_max_mantissa, self.exponent_bias)
            self.min_clamp = -self.max_clamp
            self.fp_internal_scale_min = 1. - self.exponent_bias - self.mantissa_bit_width
            self.max_value = max_float(
                self.exponent_bit_width, self.pre_compute_max_mantissa, self.exponent_bias)
            self.max_value = self.max_value if self.max_available_float is None else torch.min(
                self.max_value, self.max_available_float())
            self.min_value = torch.tensor(0.) if not module.is_signed else -self.max_value

    def quantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        # Quantize
        x = x / scale
        internal_scale = float_internal_scale(
            x, self.mantissa_bit_width, self.fp_internal_scale_min, self.eps)
        x = internal_scale * self.float_to_int_impl(x / internal_scale)

        # Compute masks
        if not self.saturating:
            inf_mask = x.isinf()
            p_max_val_mask = x > self.max_value
            n_max_val_mask = -x > self.max_value

        # Clamp
        x = self.float_clamp_impl.saturating_clamp(x, self.max_value, self.min_value)
        if not self.saturating:
            x = self.float_clamp_impl.inf_nan_clamp(x, inf_mask, p_max_val_mask, n_max_val_mask)

        return x

    def dequantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return (x - zero_point) * scale

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        return self.dequantize(self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point), self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class FloatWeightInferencetHandler(FloatInferencetHandler):
    handled_layer = WeightFloatQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.register_buffer('cached_weight', torch.ones(1))

    def prepare_for_export(self, module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            if module._cached_weight is not None and not module.cache_inference_quant_weight_metadata_only:
                self.cached_weight = module._cached_weight.value
            else:
                self.cached_weight = None

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        if self.cached_weight is not None:
            x = self.cached_weight
        else:
            x = self.inner_forward(x, self.scale, self.zero_point)
        return x, self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class GroupwiseFloatInferenceHandler(FloatInferencetHandler):
    handled_layer = GroupwiseActFloatQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant
            self.group_dim = module.group_dim

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        inp_shape = x.shape
        x, scale, zero_point, *other = self.module_forward(x)
        # If we skip quant tensor, we return the flattened version of the groupwise tensor
        x = groupwise_dequant_expand(x, scale, zero_point, self.group_dim, inp_shape)[0]
        output_args = tuple([x, scale, zero_point] + list(other))
        return output_args


class GroupwiseFloatWeightInferenceHandler(FloatWeightInferencetHandler):
    handled_layer = GroupwiseWeightFloatQuantProxyFromInjector

    def __init__(self):
        super().__init__()
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module: nn.Module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            self.input_view = module.input_view_impl
            self.group_dim = module.group_dim

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        out = self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)
        return out

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        if self.cached_weight is not None:
            out = self.cached_weight
        else:
            scale = self.scale
            if scale.shape != ():
                scale = self.input_view(scale)

            zero_point = self.zero_point
            if zero_point.shape != ():
                zero_point = self.input_view(zero_point)

            inp_shape = x.shape
            x = self.input_view(x)

            out = self.inner_forward(x, scale, zero_point)
            out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim, inp_shape)[0]

        return out, scale, zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class DynamicFloatInferenceHandler(FloatInferencetHandler):
    handled_layer = DynamicActFloatQuantProxyFromInjector

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        return self.module_forward(x)
