# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import value

from brevitas.inject import ExtendedInjector
from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.experimental.float_base import FloatActBase
from brevitas.quant.experimental.float_base import FloatWeightBase
from brevitas.quant.experimental.float_base import Fp4e2m1Mixin
from brevitas.quant.experimental.float_base import Fp6e2m3Mixin
from brevitas.quant.experimental.float_base import Fp6e3m2Mixin
from brevitas.quant.experimental.float_base import Fp8e4m3Mixin
from brevitas.quant.experimental.float_base import Fp8e5m2Mixin
from brevitas.quant.experimental.float_base import ScaledFloatActBase
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase
from brevitas.utils.float_quant_utils import get_max_available_float


class FpOCPMixin(ExtendedInjector):
    saturating = True

    @value
    def inf_values(bit_width, mantissa_bit_width, exponent_bit_width):
        if bit_width == 8:
            if mantissa_bit_width == 3 and exponent_bit_width == 4:
                return None
            if mantissa_bit_width == 2 and exponent_bit_width == 5:
                return (('00',))
        else:
            return None

    @value
    def nan_values(bit_width, mantissa_bit_width, exponent_bit_width):
        if bit_width == 8:
            if mantissa_bit_width == 3 and exponent_bit_width == 4:
                return (('111',))
            if mantissa_bit_width == 2 and exponent_bit_width == 5:
                return ('01', '11', '10')
        else:
            return None

    @value
    def max_available_float(
            exponent_bit_width, mantissa_bit_width, exponent_bias, nan_values, inf_values,
            saturating):
        return get_max_available_float(
            exponent_bit_width,
            mantissa_bit_width,
            exponent_bias,
            nan_values,
            inf_values,
            saturating)


class FpOCPWeight(FpOCPMixin, FloatWeightBase):
    """
    OCP FP8 signed weight quantizer.
    """
    pass


class FpOCPAct(FpOCPMixin, FloatActBase):
    """
    FP8 signed activation quantizer.
    """
    pass


class FpOCPWeightPerTensorFloat(FpOCPMixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_channel = False


class FpOCPActPerTensorFloat(FpOCPMixin, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_channel = False


class FpOCPWeightPerChannelFloat(FpOCPMixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_channel = True


class FpOCPActPerChannelFloat2d(FpOCPMixin, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class FpOCPActPerTensorFloatMSE(FpOCPMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_channel = False


class FpOCPActPerChannelFloat2dMSE(FpOCPMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class FpOCPWeightPerChannelFloatMSE(FpOCPMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel MSE-based scaling.
    """
    scaling_per_output_channel = True


class FpOCPWeightPerTensorFloatMSE(FpOCPMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor MSE-based scaling.
    """
    scaling_per_output_channel = False
