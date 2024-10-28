# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import this
from dependencies import value

from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.core.function_wrapper.ops_ste import FloorSte
from brevitas.core.restrict_val import PowerOfTwo
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.core.scaling.runtime import RuntimeDynamicGroupStatsScaling
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.proxy.groupwise_float_parameter_quant import \
    GroupwiseWeightFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_float_runtime_quant import GroupwiseActFloatQuantProxyFromInjector
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.proxy.groupwise_int_runtime_quant import GroupwiseActQuantProxyFromInjector
from brevitas.quant.base import IntQuant
from brevitas.quant.base import MaxStatsScaling
from brevitas.quant.base import MinMaxStatsScaling
from brevitas.quant.base import MSEAsymmetricScale
from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.base import ShiftedMinUintQuant
from brevitas.quant.experimental.float_base import ScaledFloatActBase
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase
from brevitas.quant.experimental.float_quant_ocp import FpOCPAct
from brevitas.quant.experimental.float_quant_ocp import FpOCPWeight
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.weight import WeightQuantSolver


class GroupwiseWeightFloatProxyMixin(ExtendedInjector):
    proxy_class = GroupwiseWeightFloatQuantProxyFromInjector


class GroupwiseActFloatProxyMixin(ExtendedInjector):
    proxy_class = GroupwiseActFloatQuantProxyFromInjector


class GroupwiseWeightProxyMixin(ExtendedInjector):
    proxy_class = GroupwiseWeightQuantProxyFromInjector


class GroupwiseActProxyMixin(ExtendedInjector):
    proxy_class = GroupwiseActQuantProxyFromInjector


class RestrictThresholdMixin(ExtendedInjector):
    restrict_value_float_to_int_impl = FloorSte
    restrict_scaling_impl = PowerOfTwoRestrictValue


class MXWeightMixin(ExtendedInjector):
    threshold_mixin = RestrictThresholdMixin
    group_size = 32
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = FloorSte
    scaling_per_output_type = ScalingPerOutputType.GROUP

    @value
    def restrict_threshold_impl():
        return this.threshold_mixin.restrict_scaling_impl


class MXActMixin(ExtendedInjector):
    threshold_mixin = RestrictThresholdMixin
    group_size = 32
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = FloorSte
    scaling_impl = RuntimeDynamicGroupStatsScaling
    scaling_per_output_type = ScalingPerOutputType.GROUP

    @value
    def stats_reduce_dim(group_dim):
        # If group_dim = -1, we need a workaround to avoid selecting wrong dim
        if group_dim == -1:
            return -1
        else:
            return group_dim + 1

    @value
    def restrict_threshold_impl():
        return this.threshold_mixin.restrict_scaling_impl


class MXFloat8e4m3Weight(MXWeightMixin,
                         GroupwiseWeightFloatProxyMixin,
                         FpOCPWeight,
                         ScaledFloatWeightBase):
    """
    MX Float signed weight quantizer.
    """
    bit_width = 8
    exponent_bit_width = 4
    mantissa_bit_width = 3


class MXFloat8e4m3Act(MXActMixin, GroupwiseActFloatProxyMixin, FpOCPAct, ScaledFloatActBase):
    """
    MX Float signed activation quantizer.
    """
    bit_width = 8
    exponent_bit_width = 4
    mantissa_bit_width = 3


class MXFloat8e4m3WeightMSE(MXFloat8e4m3Weight, MSESymmetricScale):
    """
    MX Float signed weight quantizer with per-channel MSE-based scaling.
    """
    pass


class MXInt8Weight(MXWeightMixin,
                   GroupwiseWeightProxyMixin,
                   IntQuant,
                   MaxStatsScaling,
                   WeightQuantSolver):
    """
    MX Int signed weight quantizer.
    """
    bit_width = 8


class ShiftedMXUInt8Weight(MXWeightMixin,
                           GroupwiseWeightProxyMixin,
                           ShiftedMinUintQuant,
                           MinMaxStatsScaling,
                           WeightQuantSolver):
    """
    MX Int signed weight quantizer.
    """
    bit_width = 8


class MXInt8Act(MXActMixin, GroupwiseActProxyMixin, IntQuant, MaxStatsScaling, ActQuantSolver):
    """
    MX Int signed activation quantizer.
    """
    bit_width = 8


class MXInt8WeightMSE(MSESymmetricScale, MXInt8Weight):
    """
    MX Int signed weight quantizer with per-channel MSE-based scaling.
    """
    pass


class ShiftedMXUInt8WeightMSE(MSEAsymmetricScale, ShiftedMXUInt8Weight):
    """
    MX Int signed weight quantizer with per-channel MSE-based scaling.
    """
    pass
