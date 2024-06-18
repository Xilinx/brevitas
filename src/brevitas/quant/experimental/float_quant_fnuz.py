# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import value

from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.experimental.float_base import FloatActBase
from brevitas.quant.experimental.float_base import FloatWeightBase
from brevitas.quant.experimental.float_base import Fp8e4m3Mixin
from brevitas.quant.experimental.float_base import Fp8e5m2Mixin
from brevitas.quant.experimental.float_base import ScaledFloatActBase
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase


class Fp8e4m3FNUZMixin(Fp8e4m3Mixin):
    nan_values = None
    inf_values = None

    @value
    def exponent_bias(exponent_bit_width):
        return 2 ** (exponent_bit_width - 1)


class Fp8e5m2FNUZMixin(Fp8e5m2Mixin):
    nan_values = None
    inf_values = None

    @value
    def exponent_bias(exponent_bit_width):
        return 2 ** (exponent_bit_width - 1)


class Fp8e4m3FNUZWeight(Fp8e4m3FNUZMixin, FloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer.
    """
    pass


class Fp8e5m2FNUZWeight(Fp8e5m2FNUZMixin, FloatWeightBase):
    """
    FP8 signed E5M2 weight quantizer.
    """
    pass


class Fp8e4m3FNUZAct(Fp8e4m3FNUZMixin, FloatActBase):
    """
    FP8 signed E4M3 activation quantizer.
    """
    pass


class Fp8e5m2FNUZAct(Fp8e5m2FNUZMixin, FloatActBase):
    """
    FP8 signed E5M2 activation quantizer.
    """
    pass


class Fp8e4m3FNUZWeightPerTensorFloat(Fp8e4m3FNUZMixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e5m2FNUZWeightPerTensorFloat(Fp8e5m2FNUZMixin, ScaledFloatWeightBase):
    """
    FP8 signed E5M2 weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e4m3FNUZActPerTensorFloat(Fp8e4m3FNUZMixin, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e5m2FNUZActPerTensorFloat(Fp8e5m2FNUZMixin, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e4m3FNUZWeightPerChannelFloat(Fp8e4m3FNUZMixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_channel = True


class Fp8e5m2FNUZWeightPerChannelFloat(Fp8e5m2FNUZMixin, ScaledFloatWeightBase):
    """
    FP8 signed E5M2 weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_channel = True


class Fp8e4m3FNUZActPerChannelFloat2d(Fp8e4m3FNUZMixin, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e5m2FNUZActPerChannelFloat2d(Fp8e5m2FNUZMixin, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e4m3FNUZActPerTensorFloatMSE(Fp8e4m3FNUZMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e5m2FNUZActPerTensorFloatMSE(Fp8e5m2FNUZMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e4m3FNUZActPerChannelFloat2dMSE(Fp8e4m3FNUZMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e5m2FNUZActPerChannelFloat2dMSE(Fp8e5m2FNUZMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e4m3FNUZWeightPerChannelFloatMSE(Fp8e4m3FNUZMixin,
                                          MSESymmetricScale,
                                          ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel MSE-based scaling.
    """
    scaling_per_output_channel = True


class Fp8e4m3FNUZWeightPerTensorFloatMSE(Fp8e4m3FNUZMixin, MSESymmetricScale,
                                         ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor MSE-based scaling.
    """
    scaling_per_output_channel = False
