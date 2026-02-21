# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from inspect import isclass
from typing import Tuple
import warnings

import torch
from torch import Tensor
import torch.nn as nn

import brevitas.config as config
from brevitas.core.function_wrapper.shape import dynamic_over_sub_channel_block_view
from brevitas.core.function_wrapper.shape import DynamicOverSubChannelBlockView
from brevitas.core.restrict_val import FloatToIntImplType
from brevitas.core.restrict_val import RestrictValueType
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
from brevitas.quant.solver.act import solve_float_to_int_impl_from_enum
from brevitas.quant.solver.common import solve_float_to_int_enum_from_impl
from brevitas.quant.solver.common import \
    solve_restrict_value_enum_from_impl  # FLOAT_TO_INT_IMPL_TO_ENUM
from brevitas.utils.quant_utils import groupwise_dequant_expand
from brevitas.utils.torch_utils import float_internal_scale


class StaticScaleZeroPointMixin(torch.nn.Module):

    def __init__(self, scale_shape=(1,), zero_point_shape=(1,)):
        self.register_buffer('scale', torch.ones(scale_shape))
        self.register_buffer('zero_point', torch.ones(zero_point_shape))

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            self.scale = module.scale_() if hasattr(module, 'scale_') else module.scale()
            self.zero_point = module.zero_point_() if hasattr(
                module, 'zero_point_') else module.zero_point()
            # Continguous is used to be extra-safe with torch.compile
            self.zero_point = self.zero_point.contiguous()
            self.scale = self.scale.contiguous()


class DynamicScaleZeroPointMixin(torch.nn.Module):

    def __init__(self):
        self.register_buffer('threshold', torch.ones(()))
        self._scaling_restriction = 'power_of_two'
        self._threshold_restriction = 'power_of_two'

    @property
    def scaling_restriction(self):
        return self._scaling_restriction

    @scaling_restriction.setter
    def scaling_restriction(self, value):
        if isclass(value):
            self._scaling_restriction = solve_restrict_value_enum_from_impl(type(value))
        elif isinstance(value, str):
            self._scaling_restriction = value
        else:
            raise "Unrecognized scaling restriction"

    @property
    def threshold_restriction(self):
        return self._threshold_restriction

    @threshold_restriction.setter
    def threshold_restriction(self, value):
        if isclass(value):
            self._threshold_restriction = solve_restrict_value_enum_from_impl(type(value))
        elif isinstance(value, str):
            self._threshold_restriction = value
        else:
            raise "Unrecognized scaling restriction"

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:

            if module.tensor_quant is not None:
                submodule = module.tensor_quant
            elif hasattr(module, 'fused_activation_quant_proxy'):
                submodule = module.fused_activation_quant_proxy.tensor_quant

            if hasattr(submodule, 'int_quant'):
                bit_width = submodule.msb_clamp_bit_width_impl()
                self.threshold = submodule.int_scaling_impl(bit_width)
            else:
                self.threshold = submodule.float_scaling_impl(
                    submodule.exponent_bit_width_impl(),
                    compute_max_mantissa(submodule.mantissa_bit_width_impl()),
                    submodule.exponent_bias_impl())

            self.scaling_restriction = submodule.scaling_impl.restrict_clamp_scaling.restrict_value_impl
            self.threshold_restriction = submodule.scaling_impl.restrict_clamp_threshold.restrict_value_impl

    # def compute_scale(self, x, group_dim):
    #     scale = torch.clamp(torch.max(torch.abs(x), dim=group_dim, keepdim=True)[0], 1e-4)
    #     threshold = self.threshold
    #     if self.scaling_restriction == RestrictValueType.POWER_OF_TWO:
    #         scale = torch.clamp(torch.pow(2, torch.floor(torch.log2(scale))), 1e-7)
    #     if self.threshold_restriction == RestrictValueType.POWER_OF_TWO:
    #         threshold = torch.clamp(torch.pow(2, torch.floor(torch.log2(threshold))), 1e-7)
    #     scale = scale / threshold
    #     return scale


class FloatToIntMixin(torch.nn.Module):

    def __init__(self):
        self._float_to_int_impl_type = 'round'

    @property
    def float_to_int_impl_type(self):
        return self._float_to_int_impl_type

    @float_to_int_impl_type.setter
    def float_to_int_impl_type(self, value):
        self._float_to_int_impl = value
        if value == FloatToIntImplType.LEARNED_ROUND:
            warnings.warn("Learned Round not supported for export, only for inference")
            return
        # A bit ugly but we need to instantiate the class
        self.float_to_int_impl = solve_float_to_int_impl_from_enum(value)()

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:

            if module.tensor_quant is not None:
                submodule = module.tensor_quant
            elif hasattr(module, 'fused_activation_quant_proxy'):
                submodule = module.fused_activation_quant_proxy.tensor_quant

            if hasattr(submodule, 'int_quant'):
                self.float_to_int_impl = submodule.int_quant.float_to_int_impl
            else:
                self.float_to_int_impl = submodule.float_to_int_impl

            # We need the class type
            self.float_to_int_impl_type = solve_float_to_int_enum_from_impl(
                type(self.float_to_int_impl))


class GroupwiseMixin(torch.nn.Module):

    def __init__(self):
        self.skip_create_quant_tensor = True
        self.register_buffer('group_dim_t', torch.ones(()))
        self.register_buffer('group_size_t', torch.ones(()))

    @property
    def group_dim(self):
        return self.group_dim_t.int()

    @property
    def group_size(self):
        return self.group_size_t.int()

    def prepare_for_export(self, module: nn.Module):
        if module.is_quant_enabled:
            self.group_dim_t = torch.tensor(module.group_dim)
            self.group_size_t = torch.tensor(module.group_size)

    def reshape(self, x, group_dim, group_size):
        init_shape = list(x.shape)
        shape = init_shape
        assert shape[group_dim] % group_size == 0
        shape[group_dim] = shape[group_dim] // group_size
        extra_dim = group_dim + 1 if group_dim != -1 else -1
        shape.insert(extra_dim, group_size)
        x = x.reshape(shape)
        return x


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

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if config.IGNORE_EXPORT_KEYS:
            return dict()
        output_dict = super(InferenceHandler, self).state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars)
        return output_dict


class IntInferencetHandlerBase(InferenceHandler, FloatToIntMixin):

    def __init__(self):
        InferenceHandler.__init__(self)
        FloatToIntMixin.__init__(self)
        self.register_buffer('bit_width', torch.ones(()))
        self.register_buffer('min_clamp', torch.ones(()))
        self.register_buffer('max_clamp', torch.ones(()))

    def prepare_for_export(self, module: nn.Module):
        InferenceHandler.prepare_for_export(self, module)
        FloatToIntMixin.prepare_for_export(self, module)
        if module.is_quant_enabled:
            self.bit_width = module.bit_width()
            self.min_clamp = min_int(module.is_signed, module.is_narrow_range, self.bit_width)
            self.max_clamp = max_int(module.is_signed, module.is_narrow_range, self.bit_width)

    def quantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        return torch.clamp(
            self.float_to_int_impl(x / scale + zero_point), self.min_clamp, self.max_clamp)

    def dequantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return (x - zero_point) * scale

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        return self.dequantize(self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point), self.scale, self.zero_point, self.bit_width


class IntInferencetHandler(IntInferencetHandlerBase, StaticScaleZeroPointMixin):
    handled_layer = (ActQuantProxyFromInjector, BiasQuantProxyFromInjector)

    def __init__(self, scale_shape=(1,), zero_point_shape=(1,)):
        IntInferencetHandlerBase.__init__(self)
        StaticScaleZeroPointMixin.__init__(self, scale_shape, zero_point_shape)

    def prepare_for_export(self, module: nn.Module):
        IntInferencetHandlerBase.prepare_for_export(self, module)
        StaticScaleZeroPointMixin.prepare_for_export(self, module)


class IntWeightInferencetHandler(IntInferencetHandler):
    handled_layer = WeightQuantProxyFromInjector

    def __init__(self, scale_shape=(1,), zero_point_shape=(1,)):
        super().__init__(scale_shape, zero_point_shape)
        self.cached_weight = None

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


class DynamicIntInferenceHandler(IntInferencetHandlerBase):
    handled_layer = DynamicActQuantProxyFromInjector

    def prepare_for_export(self, module: nn.Module):
        super().prepare_for_export(module)
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        return self.module_forward(x)


class GroupwiseIntInferenceHandler(IntInferencetHandlerBase,
                                   GroupwiseMixin,
                                   DynamicScaleZeroPointMixin):
    handled_layer = GroupwiseActQuantProxyFromInjector

    def __init__(self):
        IntInferencetHandlerBase.__init__(self)
        GroupwiseMixin.__init__(self)
        DynamicScaleZeroPointMixin.__init__(self)
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module):
        IntInferencetHandlerBase.prepare_for_export(self, module)
        GroupwiseMixin.prepare_for_export(self, module)
        DynamicScaleZeroPointMixin.prepare_for_export(self, module)
        self.module_forward = None
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant

    def forward(self, x: Tensor, unused_scale: Tensor = None) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        inp_shape = x.shape
        x, scale, zero_point, *other = self.module_forward(x)

        # If we skip quant tensor, we return the flattened version of the groupwise tensor
        x = groupwise_dequant_expand(x, scale, zero_point, self.group_dim, inp_shape)[0]
        output_args = tuple([x, scale, zero_point] + list(other))
        return output_args


class GroupwiseIntWeightInferenceHandler(IntWeightInferencetHandler, GroupwiseMixin):
    handled_layer = GroupwiseWeightQuantProxyFromInjector

    def __init__(self, scale_shape=(1,), zero_point_shape=(1,)):
        IntWeightInferencetHandler.__init__(self, scale_shape, zero_point_shape)
        GroupwiseMixin.__init__(self)
        self.skip_create_quant_tensor = True

    def prepare_for_export(self, module):
        IntWeightInferencetHandler.prepare_for_export(self, module)
        GroupwiseMixin.prepare_for_export(self, module)
        if module.is_quant_enabled:
            self.input_view = module.input_view_impl

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor

        if self.cached_weight is not None:
            out = self.cached_weight
        else:
            inp_shape = x.shape
            x = self.input_view(x)
            out = self.inner_forward(x, self.scale, self.zero_point)

            # If we skip quant tensor, we return the flattened version of the groupwise tensor
            out = groupwise_dequant_expand(
                out, self.scale, self.zero_point, self.group_dim, inp_shape)[0]
        return out, self.scale, self.zero_point, self.bit_width


class FloatInferenceHandlerBase(InferenceHandler, FloatToIntMixin):

    def __init__(self):
        InferenceHandler.__init__(self)
        FloatToIntMixin.__init__(self)
        self.register_buffer('min_clamp', torch.ones(()))
        self.register_buffer('max_clamp', torch.ones(()))
        self.register_buffer('mantissa_bit_width', torch.ones(()))
        self.register_buffer('exponent_bit_width', torch.ones(()))
        self.register_buffer('exponent_bias', torch.ones(()))
        self.register_buffer('fp_internal_scale_min', torch.ones(()))
        self.inf_values = None
        self.nan_values = None
        self.eps = 1e-8  #torch.finfo(self.scale.dtype).tiny
        self.saturating = True

    def prepare_for_export(self, module):
        InferenceHandler.prepare_for_export(self, module)
        FloatToIntMixin.prepare_for_export(self, module)
        if module.is_quant_enabled:

            self.exponent_bit_width = module.exponent_bit_width()
            self.mantissa_bit_width = module.mantissa_bit_width()
            self.exponent_bias = module.exponent_bias()
            self.saturating = module.is_saturating()
            self.inf_values = module.inf_values()
            self.nan_values = module.nan_values()
            if module.tensor_quant is not None:
                self.float_clamp_impl = module.tensor_quant.float_clamp_impl
                self.max_available_float = module.tensor_quant.float_clamp_impl.max_available_float
            elif hasattr(module, 'fused_activation_quant_proxy'):
                self.float_clamp_impl = module.fused_activation_quant_proxy.tensor_quant.float_clamp_impl
                self.max_available_float = module.fused_activation_quant_proxy.tensor_quant.float_clamp_impl.max_available_float

            self.pre_compute_max_mantissa = compute_max_mantissa(self.mantissa_bit_width)
            self.max_clamp = max_float(
                self.exponent_bit_width, self.pre_compute_max_mantissa, self.exponent_bias)
            self.max_clamp = self.max_clamp if self.max_available_float is None else torch.min(
                self.max_clamp, self.max_available_float())
            self.min_clamp = torch.tensor(0.) if not module.is_signed else -self.max_clamp

            self.fp_internal_scale_min = 1. - self.exponent_bias - self.mantissa_bit_width

    def quantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        # Quantize
        x = x / scale
        internal_scale = float_internal_scale(
            x, self.mantissa_bit_width, self.fp_internal_scale_min.to(x.device), self.eps)
        x = internal_scale * self.float_to_int_impl(x / internal_scale)

        # Compute masks
        if not self.saturating:
            inf_mask = x.isinf()
            p_max_val_mask = x > self.max_clamp
            n_max_val_mask = -x > self.max_clamp

        # Clamp
        x = torch.clamp(x, self.min_clamp.to(x.device), self.max_clamp.to(x.device))
        if not self.saturating:
            x = self.float_clamp_impl.inf_nan_clamp(x, inf_mask, p_max_val_mask, n_max_val_mask)

        return x

    def dequantize(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return (x - zero_point) * scale

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        return self.dequantize(self.quantize(x, self.scale, self.zero_point), self.scale, self.zero_point), self.scale, self.zero_point, self.exponent_bit_width, self.mantissa_bit_width, self.exponent_bias, self.saturating, self.inf_values, self.nan_values


class FloatInferencetHandler(FloatInferenceHandlerBase, StaticScaleZeroPointMixin):
    handled_layer = (ActFloatQuantProxyFromInjector, BiasQuantProxyFromInjector)

    def __init__(self, scale_shape=(1,), zero_point_shape=(1,)):
        FloatInferenceHandlerBase.__init__(self)
        StaticScaleZeroPointMixin.__init__(self, scale_shape, zero_point_shape)

    def prepare_for_export(self, module):
        FloatInferenceHandlerBase.prepare_for_export(self, module)
        StaticScaleZeroPointMixin.prepare_for_export(self, module)


class FloatWeightInferencetHandler(FloatInferencetHandler):
    handled_layer = WeightFloatQuantProxyFromInjector

    def __init__(self, scale_shape=(1,), zero_point_shape=(1,)):
        super().__init__(scale_shape, zero_point_shape)
        self.cached_weight = None

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


class GroupwiseFloatInferenceHandler(FloatInferenceHandlerBase,
                                     GroupwiseMixin,
                                     DynamicScaleZeroPointMixin):
    handled_layer = GroupwiseActFloatQuantProxyFromInjector

    def __init__(self, *args, **kwargs):
        FloatInferenceHandlerBase.__init__(self)
        GroupwiseMixin.__init__(self)
        DynamicScaleZeroPointMixin.__init__(self)

    def prepare_for_export(self, module: nn.Module):
        FloatInferenceHandlerBase.prepare_for_export(self, module)
        GroupwiseMixin.prepare_for_export(self, module)
        DynamicScaleZeroPointMixin.prepare_for_export(self, module)
        self.module_forward = None
        if module.is_quant_enabled:
            self.module_forward = module.fused_activation_quant_proxy.tensor_quant

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
        return self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        inp_shape = x.shape
        x, scale, zero_point, *other = self.module_forward(x)
        # If we skip quant tensor, we return the flattened version of the groupwise tensor
        x = groupwise_dequant_expand(x, scale, zero_point, self.group_dim, inp_shape)[0]
        output_args = tuple([x, scale, zero_point] + list(other))
        return output_args


class GroupwiseFloatWeightInferenceHandler(FloatWeightInferencetHandler, GroupwiseMixin):
    handled_layer = GroupwiseWeightFloatQuantProxyFromInjector

    def __init__(self, scale_shape=(1,), zero_point_shape=(1,)):
        FloatWeightInferencetHandler.__init__(self, scale_shape, zero_point_shape)
        GroupwiseMixin.__init__(self)
        self.skip_create_quant_tensor = True
        self.cached_weight = None

    def reshape(self, x, group_dim, group_size):
        init_shape = list(x.shape)
        shape = init_shape
        assert shape[group_dim] % group_size == 0
        shape[group_dim] = shape[group_dim] // group_size
        extra_dim = group_dim + 1 if group_dim != -1 else group_dim - 1
        shape.insert(extra_dim, group_size)
        x = x.reshape(shape)
        return x

    def prepare_for_export(self, module):
        FloatWeightInferencetHandler.prepare_for_export(self, module)
        GroupwiseMixin.prepare_for_export(self, module)

    def inner_forward(self, x: Tensor, scale: Tensor, zero_point: Tensor) -> Tuple[Tensor]:
        out = self.dequantize(self.quantize(x, scale, zero_point), scale, zero_point)
        return out

    def quantize_forward(self, x: Tensor) -> Tensor:
        if self.cached_weight is not None:
            out = self.cached_weight
        else:
            inp_shape = x.shape
            scale = self.scale
            zero_point = self.zero_point
            x = dynamic_over_sub_channel_block_view(x, self.group_size, self.group_dim)

            out = self.quantize(x, scale, zero_point)
            out = groupwise_dequant_expand(out, scale, zero_point, self.group_dim, inp_shape)[0]
        return out, None

    def forward(self, x: Tensor) -> Tuple[Tensor]:
        # In inference mode, we never return quant tensors
        assert self.skip_create_quant_tensor
        if self.cached_weight is not None:
            out = self.cached_weight
        else:
            inp_shape = x.shape
            scale = self.scale
            zero_point = self.zero_point
            x = dynamic_over_sub_channel_block_view(x, self.group_size, self.group_dim)

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
