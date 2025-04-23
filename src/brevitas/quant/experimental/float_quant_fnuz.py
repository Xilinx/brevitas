# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import value

from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.experimental.float_base import FloatActBase
from brevitas.quant.experimental.float_base import FloatWeightBase
from brevitas.quant.experimental.float_base import Fp8e4m3Mixin
from brevitas.quant.experimental.float_base import Fp8e5m2Mixin
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
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpFNUZAct(FpFNUZMixin, FloatActBase):
    """
    FP8 signed activation quantizer.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpFNUZWeightPerTensorFloat(FpFNUZMixin, ScaledFloatWeightBase):
    """
    FP8 signed weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpFNUZActPerTensorFloat(FpFNUZMixin, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpFNUZWeightPerChannelFloat(FpFNUZMixin, ScaledFloatWeightBase):
    """
    FP8 signed weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL


class FpFNUZActPerChannelFloat2d(FpFNUZMixin, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL
    scaling_stats_permute_dims = (1, 0, 2, 3)


class FpFNUZActPerTensorFloatMSE(FpFNUZMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpFNUZActPerChannelFloat2dMSE(FpFNUZMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL
    scaling_stats_permute_dims = (1, 0, 2, 3)


class FpFNUZWeightPerChannelFloatMSE(FpFNUZMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed weight quantizer with per-channel MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL


class FpFNUZWeightPerTensorFloatMSE(FpFNUZMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed weight quantizer with per-tensor MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


## Predefined FP8 Quantizers


class Fp8e4m3FNUZWeight(FpFNUZWeight, Fp8e4m3Mixin):
    """
    FNUZ FP8 E4M3 signed weight quantizer.
    """
    pass


class Fp8e4m3FNUZAct(FpFNUZAct, Fp8e4m3Mixin):
    """
    FNUZ FP8 E4M3 signed act quantizer.
    """
    pass


class Fp8e4m3FNUZWeightPerTensorFloat(FpFNUZWeightPerTensorFloat, Fp8e4m3Mixin):
    """
    FNUZ FP8 E4M3 per-tensor scaled signed weight quantizer.
    """
    pass


class Fp8e4m3FNUZWeightPerChannelFloat(FpFNUZWeightPerChannelFloat, Fp8e4m3Mixin):
    """
    FNUZ FP8 E4M3 per-channel scaled signed weight quantizer.
    """
    pass


class Fp8e4m3FNUZActPerTensorFloat(FpFNUZActPerTensorFloat, Fp8e4m3Mixin):
    """
    FNUZ FP8 E4M3 scaled signed act quantizer.
    """
    pass


class Fp8e4m3FNUZActPerTensorFloatMSE(FpFNUZActPerTensorFloatMSE, Fp8e4m3Mixin):
    """
    FNUZ FP8 E4M3 MSE-based scaled signed act quantizer.
    """
    pass


class Fp8e4m3FNUZWeightPerTensorFloatMSE(FpFNUZWeightPerTensorFloatMSE, Fp8e4m3Mixin):
    """
    FNUZ FP8 E4M3 MSE-based per-tensor scaled signed weight quantizer.
    """
    pass


class Fp8e4m3FNUZWeightPerChannelFloatMSE(FpFNUZWeightPerChannelFloatMSE, Fp8e4m3Mixin):
    """
    FNUZ FP8 E4M3 MSE-based per-channel scaled signed weight quantizer.
    """
    pass


class Fp8e5m2FNUZWeight(FpFNUZWeight, Fp8e5m2Mixin):
    """
    FNUZ FP8 e5m2 signed weight quantizer.
    """
    pass


class Fp8e5m2FNUZAct(FpFNUZAct, Fp8e5m2Mixin):
    """
    FNUZ FP8 e5m2 signed act quantizer.
    """
    pass


class Fp8e5m2FNUZWeightPerTensorFloat(FpFNUZWeightPerTensorFloat, Fp8e5m2Mixin):
    """
    FNUZ FP8 e5m2 per-tensor scaled signed weight quantizer.
    """
    pass


class Fp8e5m2FNUZWeightPerChannelFloat(FpFNUZWeightPerChannelFloat, Fp8e5m2Mixin):
    """
    FNUZ FP8 e5m2 per-channel scaled signed weight quantizer.
    """
    pass


class Fp8e5m2FNUZActPerTensorFloat(FpFNUZActPerTensorFloat, Fp8e5m2Mixin):
    """
    FNUZ FP8 e5m2 scaled signed act quantizer.
    """
    pass


class Fp8e5m2FNUZActPerTensorFloatMSE(FpFNUZActPerTensorFloatMSE, Fp8e5m2Mixin):
    """
    FNUZ FP8 e5m2 MSE-based scaled signed act quantizer.
    """
    pass


class Fp8e5m2FNUZWeightPerTensorFloatMSE(FpFNUZWeightPerTensorFloatMSE, Fp8e5m2Mixin):
    """
    FNUZ FP8 e5m2 MSE-based per-tensor scaled signed weight quantizer.
    """
    pass


class Fp8e5m2FNUZWeightPerChannelFloatMSE(FpFNUZWeightPerChannelFloatMSE, Fp8e5m2Mixin):
    """
    FNUZ FP8 e5m2 MSE-based per-channel scaled signed weight quantizer.
    """
    pass
