# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import value

from brevitas.inject import ExtendedInjector
from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.experimental.float_base import FloatActBase
from brevitas.quant.experimental.float_base import FloatWeightBase
from brevitas.quant.experimental.float_base import ScaledFloatActBase
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase


class FpFNUZMixin(ExtendedInjector):
    saturating = True

    @value
    def exponent_bias(exponent_bit_width):
        return 2 ** (exponent_bit_width - 1)


class FpFNUZWeight(FpFNUZMixin, FloatWeightBase):
    """
    FNUZ FP8 signed weight quantizer.
    """
    pass


class FpFNUZAct(FpFNUZMixin, FloatActBase):
    """
    FP8 signed activation quantizer.
    """
    pass


class FpFNUZWeightPerTensorFloat(FpFNUZMixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_channel = False


class FpFNUZActPerTensorFloat(FpFNUZMixin, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_channel = False


class FpFNUZWeightPerChannelFloat(FpFNUZMixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_channel = True


class FpFNUZActPerChannelFloat2d(FpFNUZMixin, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class FpFNUZActPerTensorFloatMSE(FpFNUZMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_channel = False


class FpFNUZActPerChannelFloat2dMSE(FpFNUZMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class FpFNUZWeightPerChannelFloatMSE(FpFNUZMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel MSE-based scaling.
    """
    scaling_per_output_channel = True


class FpFNUZWeightPerTensorFloatMSE(FpFNUZMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor MSE-based scaling.
    """
    scaling_per_output_channel = False
