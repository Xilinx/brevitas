# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.inject.enum import ScalingPerOutputType
from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.experimental.float_base import FloatActBase
from brevitas.quant.experimental.float_base import FloatWeightBase
from brevitas.quant.experimental.float_base import Fp8e4m3Mixin
from brevitas.quant.experimental.float_base import Fp8e5m2Mixin
from brevitas.quant.experimental.float_base import ScaledFloatActBase
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase


class Fp8e4m3Weight(Fp8e4m3Mixin, FloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e5m2Weight(Fp8e5m2Mixin, FloatWeightBase):
    """
    FP8 signed E5M2 weight quantizer.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e4m3Act(Fp8e4m3Mixin, FloatActBase):
    """
    FP8 signed E4M3 activation quantizer.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e5m2Act(Fp8e5m2Mixin, FloatActBase):
    """
    FP8 signed E5M2 activation quantizer.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e4m3WeightPerTensorFloat(Fp8e4m3Mixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e5m2WeightPerTensorFloat(Fp8e5m2Mixin, ScaledFloatWeightBase):
    """
    FP8 signed E5M2 weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e4m3ActPerTensorFloat(Fp8e4m3Mixin, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e5m2ActPerTensorFloat(Fp8e5m2Mixin, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e4m3WeightPerChannelFloat(Fp8e4m3Mixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL


class Fp8e5m2WeightPerChannelFloat(Fp8e5m2Mixin, ScaledFloatWeightBase):
    """
    FP8 signed E5M2 weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL


class Fp8e4m3ActPerChannelFloat2d(Fp8e4m3Mixin, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e5m2ActPerChannelFloat2d(Fp8e5m2Mixin, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e4m3ActPerTensorFloatMSE(Fp8e4m3Mixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e5m2ActPerTensorFloatMSE(Fp8e5m2Mixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class Fp8e4m3ActPerChannelFloat2dMSE(Fp8e4m3Mixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e5m2ActPerChannelFloat2dMSE(Fp8e5m2Mixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e4m3WeightPerChannelFloatMSE(Fp8e4m3Mixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL


class Fp8e4m3WeightPerTensorFloatMSE(Fp8e4m3Mixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR
