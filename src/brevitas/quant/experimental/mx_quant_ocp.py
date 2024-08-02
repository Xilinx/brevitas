# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import value

from brevitas.core.function_wrapper.ops_ste import CeilSte
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
from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.experimental.float_base import ScaledFloatActBase
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase
from brevitas.quant.experimental.float_quant_ocp import FpOCPAct
from brevitas.quant.experimental.float_quant_ocp import FpOCPWeight
from brevitas.quant.solver.act import ActQuantSolver
from brevitas.quant.solver.weight import WeightQuantSolver


class MXFloatWeightMixin(ExtendedInjector):
    proxy_class = GroupwiseWeightFloatQuantProxyFromInjector
    group_size = 32
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte
    scaling_per_output_type = ScalingPerOutputType.GROUP


class MXFloatActMixin(ExtendedInjector):
    proxy_class = GroupwiseActFloatQuantProxyFromInjector
    group_size = 32
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte
    scaling_impl = RuntimeDynamicGroupStatsScaling
    scaling_per_output_type = ScalingPerOutputType.GROUP

    @value
    def stats_reduce_dim(group_dim):
        # If group_dim = -1, we need a workaround to avoid selecting wrong dim
        if group_dim == -1:
            return -1
        else:
            return group_dim + 1


class MXIntWeightMixin(ExtendedInjector):
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    group_size = 32
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte
    scaling_per_output_type = ScalingPerOutputType.GROUP


class MXIntActMixin(ExtendedInjector):
    proxy_class = GroupwiseActQuantProxyFromInjector
    group_size = 32
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte
    scaling_impl = RuntimeDynamicGroupStatsScaling
    scaling_per_output_type = ScalingPerOutputType.GROUP

    @value
    def stats_reduce_dim(group_dim):
        # If group_dim = -1, we need a workaround to avoid selecting wrong dim
        if group_dim == -1:
            return -1
        else:
            return group_dim + 1


class MXFloatWeight(MXFloatWeightMixin, FpOCPWeight, ScaledFloatWeightBase):
    """
    MX Float signed weight quantizer.
    """
    pass


class MXFloatAct(MXFloatActMixin, FpOCPAct, ScaledFloatActBase):
    """
    MX Float signed activation quantizer.
    """
    pass


class MXFloatWeightMSE(MXFloatWeight, MSESymmetricScale):
    """
    MX Float signed weight quantizer with per-channel MSE-based scaling.
    """
    pass


class MXIntWeight(MXIntWeightMixin, IntQuant, MaxStatsScaling, WeightQuantSolver):
    """
    MX Int signed weight quantizer.
    """
    pass


class MXIntAct(MXIntActMixin, IntQuant, MaxStatsScaling, ActQuantSolver):
    """
    MX Int signed activation quantizer.
    """
    pass


class MXIntWeightMSE(MXIntWeight, MSESymmetricScale):
    """
    MX Int signed weight quantizer with per-channel MSE-based scaling.
    """
    pass
