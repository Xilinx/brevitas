# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import this
from dependencies import value
from torch import nn

from brevitas.core.bit_width import BitWidthConst
from brevitas.core.bit_width import BitWidthStatefulConst
from brevitas.core.function_wrapper import OverOutputChannelView
from brevitas.core.function_wrapper import RoundToZeroSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.function_wrapper import TensorClampSte
from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.core.function_wrapper.shape import StatsInputViewShapeImpl
from brevitas.core.quant import ClampedBinaryQuant
from brevitas.core.quant.int import DecoupledRescalingIntQuant
from brevitas.core.quant.int import DecoupledRescalingIntQuantWithInput
from brevitas.core.quant.int_base import DecoupledIntQuant
from brevitas.core.restrict_val import FloatRestrictValue
from brevitas.core.restrict_val import LogFloatRestrictValue
from brevitas.core.scaling import AccumulatorAwareParameterPreScaling
from brevitas.core.scaling import AccumulatorAwareZeroCenterParameterPreScaling
from brevitas.core.scaling import IntScaling
from brevitas.core.scaling import ParameterFromStatsFromParameterScaling
from brevitas.core.scaling import ParameterPreScalingWeightNorm
from brevitas.core.scaling import ParameterScaling
from brevitas.core.scaling import SCALAR_SHAPE
from brevitas.core.scaling import SCALING_STATS_REDUCE_DIM
from brevitas.core.scaling import StatsFromParameterScaling
from brevitas.core.stats import AbsMax
from brevitas.core.stats import AbsMaxL2
from brevitas.core.stats import AbsMinMax
from brevitas.core.stats import L1Norm
from brevitas.core.stats import L2Norm
from brevitas.core.stats import MSE
from brevitas.core.stats import NegativeMinOrZero
from brevitas.core.stats import NegativePercentileOrZero
from brevitas.core.utils import SingleArgStatelessBuffer
from brevitas.core.zero_point import ParameterFromRuntimeZeroPoint
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.core.zero_point import PreZeroCenterZeroPoint
from brevitas.core.zero_point import StatsFromParameterZeroPoint
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import BitWidthImplType
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import StatsOp
from brevitas.proxy import DecoupledWeightQuantProxyFromInjector
from brevitas.proxy import DecoupledWeightQuantWithInputProxyFromInjector
from brevitas.quant.solver.parameter import SolveParameterScalingShape
from brevitas.quant.solver.weight import SolveWeightScalingPerOutputChannelShapeFromModule
from brevitas.quant.solver.weight import SolveWeightScalingStatsInputDimsFromModule

__all__ = [
    'MaxStatsScaling',
    'MinMaxStatsScaling',
    'ParamFromRuntimePercentileScaling',
    'ParamFromRuntimePercentileIntervalScaling',
    'ParamFromRuntimeMinMaxScaling',
    'ParamMinMaxInitScaling',
    'IntQuant',
    'NarrowIntQuant',
    'UintQuant',
    'ShiftedMinUintQuant',
    'ShiftedParamFromPercentileUintQuant',
    'PerChannelFloatScaling8bit',
    'PerTensorFloatScaling8bit',
    'PerTensorPoTScaling8bit',
    'PerChannelPoTScaling8bit',
    'SignedBinaryClampedConst',
    'WeightPerTensorFloatDecoupledL2Param',
    'WeightPerChannelFloatDecoupled',
    'WeightNormPerChannelFloatDecoupled',
    'BatchQuantStatsScaling1d',
    'BatchQuantStatsScaling2d',
    'AccumulatorAwareWeightQuant',
    'AccumulatorAwareZeroCenterWeightQuant',
    'MSESymmetricScale',
    'MSEAsymmetricScale',
    'MSEWeightZeroPoint',
    'MSEActZeroPoint']


class MaxStatsScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX
    scaling_min_val = 1e-10


class MinMaxStatsScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MIN_MAX
    scaling_min_val = 1e-10


class ParamFromRuntimePercentileScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.PERCENTILE
    high_percentile_q = 99.999
    collect_stats_steps = 300
    scaling_min_val = 1e-10


class BatchQuantStatsScaling1d(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.AFFINE_STATS
    scaling_stats_op = StatsOp.MAX_AVE
    scaling_stats_momentum = None  # perform a true average rather than exp average
    affine_shift_scale = False  # max_ave statistics are scaled but not shifted
    scaling_stats_permute_dims = (1, 0, 2)  # assuming (N, C, F), put channel dimension first
    scaling_min_val = 1e-10


class BatchQuantStatsScaling2d(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.AFFINE_STATS
    scaling_stats_op = StatsOp.MAX_AVE
    scaling_stats_momentum = None  # perform a true average rather than exp average
    affine_shift_scale = False  # max_ave statistics are scaled but not shifted
    scaling_stats_permute_dims = (1, 0, 2, 3)  # assuming (N, C, H, W), put channel dimension first
    scaling_min_val = 1e-10


class ParamFromRuntimeMinMaxScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.MIN_MAX
    collect_stats_steps = 300
    scaling_min_val = 1e-10


class ParamFromRuntimePercentileIntervalScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_op = StatsOp.PERCENTILE_INTERVAL
    high_percentile_q = 99.999
    low_percentile_q = 0.001
    collect_stats_steps = 300
    scaling_min_val = 1e-10


class ParamMinMaxInitScaling(ExtendedInjector):
    """
    """
    scaling_impl_type = ScalingImplType.PARAMETER


class IntQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = True
    zero_point_impl = ZeroZeroPoint


class NarrowIntQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = True
    signed = True
    zero_point_impl = ZeroZeroPoint


class UintQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    zero_point_impl = ZeroZeroPoint


class ShiftedMinUintQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    quantize_zero_point = True
    zero_point_impl = StatsFromParameterZeroPoint
    zero_point_stats_impl = NegativeMinOrZero
    zero_point_shape = this.scaling_shape
    zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl
    zero_point_stats_input_concat_dim = this.scaling_stats_input_concat_dim


class ShiftedParamFromPercentileUintQuant(ExtendedInjector):
    """
    """
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    float_to_int_impl_type = FloatToIntImplType.ROUND
    narrow_range = False
    signed = False
    quantize_zero_point = True
    zero_point_impl = ParameterFromRuntimeZeroPoint
    zero_point_stats_impl = NegativePercentileOrZero
    low_percentile_q = 0.001
    zero_point_shape = this.scaling_shape
    zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl


class PerChannelFloatScaling8bit(ExtendedInjector):
    """
    """
    scaling_per_output_channel = True
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class PerTensorFloatScaling8bit(ExtendedInjector):
    """
    """
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class PerChannelPoTScaling8bit(ExtendedInjector):
    """
    """
    scaling_per_output_channel = True
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8


class PerTensorPoTScaling8bit(ExtendedInjector):
    """
    """
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    bit_width = 8
    restrict_value_float_to_int_impl = CeilSte


class SignedBinaryClampedConst(ExtendedInjector):
    tensor_quant = ClampedBinaryQuant
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    float_to_int_impl_type = FloatToIntImplType.ROUND
    scaling_per_output_channel = False
    narrow_range = True
    signed = True


class PerTensorConstScaling2bit(ExtendedInjector):
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    scaling_per_output_channel = False
    bit_width = 2


class WeightPerTensorFloatDecoupledL2Param(SolveWeightScalingStatsInputDimsFromModule):
    """
    Experimental narrow per-tensor signed int weight quantizer fragment with decoupled L2,inf
    normalization and learned scaling.
    """

    proxy_class = DecoupledWeightQuantProxyFromInjector
    tensor_quant = DecoupledRescalingIntQuant
    decoupled_int_quant = DecoupledIntQuant
    tensor_clamp_impl = TensorClampSte
    pre_scaling_impl = StatsFromParameterScaling
    scaling_stats_impl = AbsMaxL2
    scaling_stats_input_view_shape_impl = OverOutputChannelView
    scaling_stats_input_concat_dim = 0
    stats_reduce_dim = SCALING_STATS_REDUCE_DIM
    restrict_scaling_impl = FloatRestrictValue
    scaling_shape = SCALAR_SHAPE
    scaling_impl = ParameterFromStatsFromParameterScaling
    int_scaling_impl = IntScaling
    zero_point_impl = ZeroZeroPoint
    pre_zero_point_impl = ZeroZeroPoint
    bit_width_impl = BitWidthConst
    narrow_range = True
    signed = True


class WeightPerChannelFloatDecoupled(SolveWeightScalingStatsInputDimsFromModule,
                                     SolveWeightScalingPerOutputChannelShapeFromModule,
                                     SolveParameterScalingShape):
    """
    Experimental narrow per-channel signed int weight quantizer fragment with decoupled Linf
    normalization and learned scaling.
    """

    proxy_class = DecoupledWeightQuantProxyFromInjector
    tensor_quant = DecoupledRescalingIntQuant
    decoupled_int_quant = DecoupledIntQuant
    tensor_clamp_impl = TensorClampSte
    pre_scaling_impl = StatsFromParameterScaling
    scaling_stats_impl = AbsMax
    restrict_scaling_impl = FloatRestrictValue
    scaling_impl = ParameterFromStatsFromParameterScaling
    int_scaling_impl = IntScaling
    zero_point_impl = ZeroZeroPoint
    pre_zero_point_impl = ZeroZeroPoint
    bit_width_impl = BitWidthConst
    narrow_range = True
    signed = True
    scaling_stats_input_view_shape_impl = OverOutputChannelView
    stats_reduce_dim = SCALING_STATS_REDUCE_DIM
    scaling_per_output_channel = True


class WeightNormPerChannelFloatDecoupled(SolveWeightScalingStatsInputDimsFromModule,
                                         SolveWeightScalingPerOutputChannelShapeFromModule,
                                         SolveParameterScalingShape):
    """Experimental narrow per-channel weight normalization-based signed integer quantizer
    based on `Quantized Neural Networks for Low-Precision Accumulation with Guaranteed
    Overflow Avoidance` by I. Colbert, A. Pappalardo, and J. Petri-Koenig.

    The formulation for weight normalization-based quantization is given below:
        `y = clip(round( (g / s) * (w / norm(w)) )) * s`

    The default quantizer uses the decoupled rescaling integer quantization arithmetic
    where the weight normalization calculation and parameterization are combined with the
    scaling factor to become the pre-clipping scaling factor (i.e., `pre_scale`) and the
    scaling factor is the post-clipping scaling factor (i.e., `post_scale`). For further
    details on the arithmetic, see `ParameterPreScalingWeightNorm`. For further details
    on the weight normalization-based quantization technique, see the referenced paper."""

    @value
    def scaling_init(scaling_init_impl, bit_width):
        scales = scaling_init_impl.parameter_list_stats() / (pow(2., bit_width - 1.) - 1.)
        return scales

    proxy_class = DecoupledWeightQuantProxyFromInjector
    tensor_quant = DecoupledRescalingIntQuant
    decoupled_int_quant = DecoupledIntQuant
    tensor_clamp_impl = TensorClamp
    scaling_impl = ParameterScaling
    scaling_init_impl = StatsFromParameterScaling
    restrict_scaling_impl = LogFloatRestrictValue
    scaling_stats_impl = AbsMax
    pre_scaling_impl = ParameterPreScalingWeightNorm
    restrict_pre_scaling_impl = LogFloatRestrictValue
    normalize_stats_impl = L2Norm
    pre_scaling_shape = this.scaling_shape  # TODO: decouple pre_scaling_shape from scaling_shape
    int_scaling_impl = SingleArgStatelessBuffer(1.)
    zero_point_impl = ZeroZeroPoint
    pre_zero_point_impl = ZeroZeroPoint
    bit_width_impl = BitWidthConst
    narrow_range = True
    signed = True
    scaling_stats_input_view_shape_impl = OverOutputChannelView
    stats_reduce_dim = SCALING_STATS_REDUCE_DIM
    scaling_per_output_channel = True
    scaling_min_val = 1e-10
    pre_scaling_min_val = 1e-10


class AccumulatorAwareWeightQuant(WeightNormPerChannelFloatDecoupled):
    """Experimental accumulator-aware weight quantizer based on `Quantized Neural Networks
    for Low-Precision Accumulation with Guaranteed Overflow Avoidance` by I. Colbert,
    A. Pappalardo, and J. Petri-Koenig.

    The formulation is based on weight normalization-based quantization as given below:
        `y = clip(round( (g / s) * (w / norm(w)) )) * s`
    where `g` is the constrained such that the upper-bound on the l1-norm of the quantized
    weights satisfies the bounds derived in the referenced paper.

    The default quantizer uses input-aware decoupled rescaling integer quantization arithmetic
    where the constrained weight normalization calculation and parameterization are combined
    with the scaling factor to become the pre-clipping scaling factor (i.e., `pre_scale`) an
    the scaling factor is the post-clipping scaling factor (i.e., `post_scale`). For further
    details on the arithmetic, see `AccumulatorAwareParameterPreScalingWeightNorm`. For further
    details on accumulator-aware quantization (A2Q) technique, see the referenced paper."""

    @value
    def accumulator_bit_width_impl(accumulator_bit_width):
        return BitWidthStatefulConst(accumulator_bit_width)

    proxy_class = DecoupledWeightQuantWithInputProxyFromInjector
    tensor_quant = DecoupledRescalingIntQuantWithInput
    pre_scaling_impl = AccumulatorAwareParameterPreScaling
    accumulator_bit_width = 32  # default maximum accumulator width is 32 bits
    normalize_stats_impl = L1Norm  # required to align with derivations in paper
    float_to_int_impl = RoundToZeroSte  # required to ensure no upwards rounding violates constraints


class AccumulatorAwareZeroCenterWeightQuant(AccumulatorAwareWeightQuant):
    """Experimental zero-centered accumulator-aware weight quantized based on:
    `A2Q+: Improving Accumulator-Aware Weight Quantization`.

    When compared to A2Q, A2Q+ changes the following:
    (1) added zero-centering constraint on the weights (i.e., `PreZeroCenterZeroPoint`)
    (2) a more relaxed l1-norm bound that is derived in the referenced paper
    """
    pre_scaling_impl = AccumulatorAwareZeroCenterParameterPreScaling
    pre_zero_point_impl = PreZeroCenterZeroPoint
    pre_zero_point_shape = this.scaling_shape  # TODO: decouple zero_point from scaling
    pre_zero_point_stats_input_view_shape_impl = this.scaling_stats_input_view_shape_impl


class MSESubInjectorBase(ExtendedInjector):

    @value
    def inner_stats_input_view_shape_impl(per_channel):
        if per_channel:
            return StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
        else:
            return StatsInputViewShapeImpl.OVER_TENSOR

    permute_dims = (this << 1).permute_dims


class MSESymmetricScaleSubInjector(MSESubInjectorBase):
    per_channel = (this << 1).scaling_per_output_channel
    proxy_module = (this << 1).proxy_module
    mse_init_op = AbsMax
    stats_impl = MSE
    stats_reduce_dim = (this << 1).stats_reduce_dim


class MSEAsymmetricScaleSubInjector(MSESubInjectorBase):
    per_channel = (this << 1).scaling_per_output_channel
    proxy_module = (this << 1).proxy_module
    mse_init_op = AbsMinMax
    stats_impl = MSE
    stats_reduce_dim = (this << 1).stats_reduce_dim


class MSEZeroPointSubInjector(MSESubInjectorBase):
    # zp is per channel when scaling is per channel
    per_channel = (this << 1).scaling_per_output_channel
    proxy_module = (this << 1).proxy_module
    mse_init_op = NegativeMinOrZero
    mse_search_method = 'grid'
    stats_impl = MSE
    stats_reduce_dim = (this << 1).stats_reduce_dim


class MSEAsymmetricScale(ExtendedInjector):
    """
    We leverage a sub-injector to avoid a name clash between scale and zero-point.
    """

    mse_scale = MSEAsymmetricScaleSubInjector
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_input_view_shape_impl = nn.Identity()

    @value
    def scaling_stats_impl():
        return this.mse_scale.stats_impl


class MSESymmetricScale(ExtendedInjector):
    """
    We leverage a sub-injector to avoid a name clash between scale and zero-point.
    """

    mse_scale = MSESymmetricScaleSubInjector
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    scaling_stats_input_view_shape_impl = nn.Identity()

    @value
    def scaling_stats_impl():
        return this.mse_scale.stats_impl


class MSEZeroPoint(ExtendedInjector):
    """
    We leverage a sub-injector to avoid a name clash between scale and zero-point.
    """

    mse_zero_point = MSEZeroPointSubInjector
    zero_point_stats_input_view_shape_impl = nn.Identity()

    @value
    def zero_point_stats_impl():
        return this.mse_zero_point.stats_impl


class MSEWeightZeroPoint(MSEZeroPoint):
    zero_point_impl = ParameterFromStatsFromParameterZeroPoint


class MSEActZeroPoint(MSEZeroPoint):
    zero_point_impl = ParameterFromRuntimeZeroPoint
