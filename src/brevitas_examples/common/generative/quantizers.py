"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from torch import nn

from brevitas.core.function_wrapper.ops_ste import FloorSte
from brevitas.core.function_wrapper.shape import OverOutputFeaturesView
from brevitas.core.function_wrapper.shape import OverTensorView
from brevitas.core.quant.float import FloatQuant
from brevitas.core.restrict_val import QuantRestrictValue
from brevitas.core.scaling.runtime import RuntimeDynamicGroupStatsScaling
from brevitas.core.stats import AbsMinMax
from brevitas.core.stats import NegativeMinOrZero
from brevitas.core.stats.stats_op import HalfQuadraticOptimizerZeroPoint
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.core.zero_point import RuntimeDynamicGroupZeroPoint
from brevitas.core.zero_point import StatsFromParameterZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.inject import this
from brevitas.inject import value
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.proxy.float_runtime_quant import DynamicActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_runtime_quant import GroupwiseActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.groupwise_int_runtime_quant import GroupwiseActQuantProxyFromInjector
from brevitas.proxy.runtime_quant import DynamicActQuantProxyFromInjector
from brevitas.quant.base import HQOWeightZeroPoint
from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.base import PerChannelPoTScaling8bit
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3WeightPerChannelFloat
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloat
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat

from .quant_blocks import *


class DynamicActProxyMixin(ExtendedInjector):
    proxy_class = DynamicActQuantProxyFromInjector


class IntWeightSymmetricGroupQuant(Int8WeightPerChannelFloat):
    """
    Block / group / vector signed symmetric int weight quantizer with float scales.
    We inherit from a per-channel quantizer to re-use some underlying machinery.
    """
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP


class Fp8e4m3WeightSymmetricGroupQuant(Fp8e4m3WeightPerChannelFloat):
    """
    Block / group / vector signed symmetric e4m3 weight quantizer with float scales.
    We inherit from a per-channel quantizer to re-use some underlying machinery.
    """
    proxy_class = GroupwiseWeightFloatQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP


class Int8DynamicActPerTensorFloat(DynamicActProxyMixin, Int8ActPerTensorFloat):
    """
    Symmetric quantizer with per tensor dynamic scale.
    """
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverTensorView
    scaling_stats_op = 'min_max'
    dynamic_scaling_broadcastable_fn = lambda x, shape: x.view(SCALAR_SHAPE)


class Fp8e4m3FNUZDynamicActPerTensorFloat(Fp8e4m3FNUZActPerTensorFloat):
    """
    Symmetric quantizer with per tensor dynamic scale.
    """
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverTensorView
    scaling_stats_op = 'min_max'
    dynamic_scaling_broadcastable_fn = lambda x, shape: x.view(SCALAR_SHAPE)


class Int8DynamicActPerRowFloat(DynamicActProxyMixin, Int8ActPerTensorFloat):
    """
    Symmetric quantizer with per row dynamic scale.
    """
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverOutputFeaturesView
    scaling_stats_op = 'min_max'
    scaling_per_output_channel = True


class Int8DynamicActPerRowFixedPoint(Int8DynamicActPerRowFloat):
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = FloorSte


class Int8DynamicActPerGroupFloat(DynamicActProxyMixin, Int8ActPerTensorFloat):
    """
    Symmetric quantizer with per group scale.
    """
    proxy_class = GroupwiseActQuantProxyFromInjector
    scaling_impl = RuntimeDynamicGroupStatsScaling
    scaling_stats_op = 'min_max'
    scaling_per_output_type = ScalingPerOutputType.GROUP


class ShiftedUint8DynamicActPerGroupFloat(DynamicActProxyMixin, ShiftedUint8ActPerTensorFloat):
    """
    Symmetric quantizer with per group scale.
    """
    proxy_class = GroupwiseActQuantProxyFromInjector
    scaling_impl = RuntimeDynamicGroupStatsScaling
    scaling_stats_op = 'min_max'
    scaling_per_output_type = ScalingPerOutputType.GROUP
    zero_point_impl = RuntimeDynamicGroupZeroPoint
    zero_point_stats_impl = NegativeMinOrZero


class ShiftedUint8DynamicActPerTensorFloat(DynamicActProxyMixin, ShiftedUint8ActPerTensorFloat):
    """
    Symmetric quantizer with per tensor dynamic scale.
    """
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverTensorView
    scaling_stats_op = 'min_max'
    zero_point_impl = RuntimeDynamicStatsZeroPoint
    zero_point_stats_impl = NegativeMinOrZero
    dynamic_scaling_broadcastable_fn = lambda x, shape: x.view(SCALAR_SHAPE)


class ShiftedUint8DynamicActPerRowFloat(DynamicActProxyMixin, ShiftedUint8ActPerTensorFloat):
    """
    Asymmetric quantizer with per row dynamic scale.
    """
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverOutputFeaturesView
    scaling_stats_op = 'min_max'
    scaling_per_output_channel = True
    zero_point_impl = RuntimeDynamicStatsZeroPoint
    zero_point_stats_impl = NegativeMinOrZero


class Fp8e4m3DynamicActPerGroupFloat(DynamicActProxyMixin, Fp8e4m3ActPerTensorFloat):
    """
    Symmetric quantizer with per group scale.
    """
    proxy_class = GroupwiseActFloatQuantProxyFromInjector
    scaling_impl = RuntimeDynamicGroupStatsScaling
    scaling_per_output_type = ScalingPerOutputType.GROUP
    scaling_stats_op = 'min_max'


class FP8e4m3OCPDynamicActPerRowFixedPoint(Fp8e4m3OCPActPerTensorFloat):
    """
    Symmetric quantizer with per row dynamic scale.
    """
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverOutputFeaturesView
    scaling_stats_op = 'min_max'
    scaling_per_output_channel = True
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = FloorSte
    proxy_class = DynamicActFloatQuantProxyFromInjector


class FP8e4m3OCPDynamicActPerRowFloat(Fp8e4m3OCPActPerTensorFloat):
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverOutputFeaturesView
    scaling_stats_op = 'min_max'
    scaling_per_output_channel = True
    proxy_class = DynamicActFloatQuantProxyFromInjector


class Fp8e4m3OCPDynamicActPerGroupFloat(DynamicActProxyMixin, Fp8e4m3OCPActPerTensorFloat):
    """
    Symmetric quantizer with per group scale.
    """
    proxy_class = GroupwiseActFloatQuantProxyFromInjector
    scaling_impl = RuntimeDynamicGroupStatsScaling
    scaling_per_output_type = ScalingPerOutputType.GROUP
    scaling_stats_op = 'min_max'


class Fp8e4m3OCPWeightSymmetricGroupQuant(Fp8e4m3OCPWeightPerChannelFloat):
    """
    Block / group / vector signed symmetric e4m3 OCP weight quantizer with float scales.
    We inherit from a per-channel quantizer to re-use some underlying machinery.
    """
    proxy_class = GroupwiseWeightFloatQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP


class Fp8e4m3OCPWeightPerChannelFixedPointMSE(MSESymmetricScale,
                                              PerChannelPoTScaling8bit,
                                              Fp8e4m3OCPWeightPerChannelFloat):
    pass


class Fp8e4m3OCPWeightPerChannelFloatMSE(MSESymmetricScale, Fp8e4m3OCPWeightPerChannelFloat):
    pass


class FP8e4m3FNUZDynamicActPerRowFloat(Fp8e4m3FNUZActPerTensorFloat):
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverOutputFeaturesView
    scaling_stats_op = 'min_max'
    scaling_per_output_channel = True
    proxy_class = DynamicActFloatQuantProxyFromInjector


class ConstActQuantScalingFloat(QuantScaleScaleShapeMixin,
                               Fp8e4m3OCPActPerTensorFloat):
    module = (this << 1).module
    upstream_scaling = (this << 1).scaling_per_output_type
    scaling_impl_type = "const"
    scaling_init = 1.0


class DynamicActQuantScalingFloat(QuantScaleScaleShapeMixin,
                               DynamicActProxyMixin,
                               Fp8e4m3OCPActPerTensorFloat):
    module = (this << 1).module
    upstream_scaling = (this << 1).scaling_per_output_type
    scaling_impl = RuntimeDynamicStatsScaling
    scaling_stats_input_view_shape_impl = OverTensorView
    scaling_stats_op = 'min_max'
    dynamic_scaling_broadcastable_fn = lambda x, shape: x.view(SCALAR_SHAPE)


class StaticActQuantScalingFloat(QuantScaleScaleShapeMixin,
                               DynamicActProxyMixin,
                               Fp8e4m3OCPActPerTensorFloat):
    module = (this << 1).module
    upstream_scaling = (this << 1).scaling_per_output_type
    scaling_stats_op = 'min_max'


class DynamicQuantScaleMXFloat8e4m3Act(MXFloat8e4m3Act):
    scaling_float_quant = StaticActQuantScalingFloat
    restrict_scaling_impl = QuantRestrictValue

    @value
    def restrict_value_float_to_int_impl():
        return this.scaling_float_quant.tensor_quant

    @value
    def scale_dequantized_shape(scaling_per_output_type, scaling_shape):
        if scaling_per_output_type == ScalingPerOutputType.TENSOR or scaling_per_output_type == ScalingPerOutputType.CHANNEL:
            return None
        elif scaling_per_output_type == ScalingPerOutputType.GROUP:
            return scaling_shape


class ConstQuantWeightScalingFloat(QuantScaleScaleShapeMixin, Fp8e4m3OCPWeightPerTensorFloat):
    module = (this << 1).module
    tracked_parameter_list = (this << 1).tracked_parameter_list
    upstream_scaling = (this << 1).scaling_per_output_type
    scaling_impl_type = "const"
    scaling_init = 1.0


class QuantWeightScalingFloat(QuantScaleScaleShapeMixin, Fp8e4m3OCPWeightPerTensorFloat):
    module = (this << 1).module
    tracked_parameter_list = (this << 1).tracked_parameter_list
    upstream_scaling = (this << 1).scaling_per_output_type
    float_quant = FloatQuant


class QuantScaleMXFloat8e4m3Weight(MXFloat8e4m3Weight):
    scaling_float_quant = QuantWeightScalingFloat
    restrict_scaling_impl = QuantRestrictValue

    @value
    def restrict_value_float_to_int_impl():
        return this.scaling_float_quant.tensor_quant

    @value
    def scale_dequantized_shape(scaling_per_output_type, scaling_shape):
        if scaling_per_output_type == ScalingPerOutputType.TENSOR or scaling_per_output_type == ScalingPerOutputType.CHANNEL:
            return None
        elif scaling_per_output_type == ScalingPerOutputType.GROUP:
            return scaling_shape


class QuantScaleMXFloat8e4m3WeightMSE(MSESymmetricScale, MXFloat8e4m3Weight):
    scaling_float_quant = QuantWeightScalingFloat
    restrict_scaling_impl = QuantRestrictValue

    @value
    def restrict_value_float_to_int_impl():
        return this.scaling_float_quant.tensor_quant

    @value
    def scale_dequantized_shape(scaling_per_output_type, scaling_shape):
        if scaling_per_output_type == ScalingPerOutputType.TENSOR or scaling_per_output_type == ScalingPerOutputType.CHANNEL:
            return None
        elif scaling_per_output_type == ScalingPerOutputType.GROUP:
            return scaling_shape
