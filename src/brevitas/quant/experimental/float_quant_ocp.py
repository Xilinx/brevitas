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
    OCP FP signed weight quantizer.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpOCPAct(FpOCPMixin, FloatActBase):
    """
    OCP FP signed activation quantizer.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpOCPWeightPerTensorFloat(FpOCPMixin, ScaledFloatWeightBase):
    """
    OCP FP signed E3M4 weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpOCPActPerTensorFloat(FpOCPMixin, ScaledFloatActBase):
    """
    OCP FP signed activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpOCPWeightPerChannelFloat(FpOCPMixin, ScaledFloatWeightBase):
    """
    OCP FP signed E3M4 weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL


class FpOCPActPerChannelFloat2d(FpOCPMixin, ScaledFloatActBase):
    """
    OCP FP signed activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL
    scaling_stats_permute_dims = (1, 0, 2, 3)


class FpOCPActPerTensorFloatMSE(FpOCPMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    OCP FP signed activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


class FpOCPActPerChannelFloat2dMSE(FpOCPMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    OCP FP signed activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL
    scaling_stats_permute_dims = (1, 0, 2, 3)


class FpOCPWeightPerChannelFloatMSE(FpOCPMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    OCP FP signed E3M4 weight quantizer with per-channel MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.CHANNEL


class FpOCPWeightPerTensorFloatMSE(FpOCPMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    OCP FP signed E3M4 weight quantizer with per-tensor MSE-based scaling.
    """
    scaling_per_output_type = ScalingPerOutputType.TENSOR


## Predefined FP8 Quantizers


class Fp8e4m3OCPWeight(FpOCPWeight, Fp8e4m3Mixin):
    """
    OCP FP8 E4M3 signed weight quantizer.
    """
    pass


class Fp8e4m3OCPAct(FpOCPAct, Fp8e4m3Mixin):
    """
    OCP FP8 E4M3 signed act quantizer.
    """
    pass


class Fp8e4m3OCPWeightPerTensorFloat(FpOCPWeightPerTensorFloat, Fp8e4m3Mixin):
    """
    OCP FP8 E4M3 per-tensor scaled signed weight quantizer.
    """
    pass


class Fp8e4m3OCPWeightPerChannelFloat(FpOCPWeightPerChannelFloat, Fp8e4m3Mixin):
    """
    OCP FP8 E4M3 per-channel scaled signed weight quantizer.
    """
    pass


class Fp8e4m3OCPActPerTensorFloat(FpOCPActPerTensorFloat, Fp8e4m3Mixin):
    """
    OCP FP8 E4M3 scaled signed act quantizer.
    """
    pass


class Fp8e4m3OCPActPerTensorFloatMSE(FpOCPActPerTensorFloatMSE, Fp8e4m3Mixin):
    """
    OCP FP8 E4M3 MSE-based scaled signed act quantizer.
    """
    pass


class Fp8e4m3OCPWeightPerTensorFloatMSE(FpOCPWeightPerTensorFloatMSE, Fp8e4m3Mixin):
    """
    OCP FP8 E4M3 MSE-based per-tensor scaled signed weight quantizer.
    """
    pass


class Fp8e4m3OCPWeightPerChannelFloatMSE(FpOCPWeightPerChannelFloatMSE, Fp8e4m3Mixin):
    """
    OCP FP8 E4M3 MSE-based per-channel scaled signed weight quantizer.
    """
    pass


class Fp8e5m2OCPWeight(FpOCPWeight, Fp8e5m2Mixin):
    """
    OCP FP8 e5m2 signed weight quantizer.
    """
    pass


class Fp8e5m2OCPAct(FpOCPAct, Fp8e5m2Mixin):
    """
    OCP FP8 e5m2 signed act quantizer.
    """
    pass


class Fp8e5m2OCPWeightPerTensorFloat(FpOCPWeightPerTensorFloat, Fp8e5m2Mixin):
    """
    OCP FP8 e5m2 per-tensor scaled signed weight quantizer.
    """
    pass


class Fp8e5m2OCPWeightPerChannelFloat(FpOCPWeightPerChannelFloat, Fp8e5m2Mixin):
    """
    OCP FP8 e5m2 per-channel scaled signed weight quantizer.
    """
    pass


class Fp8e5m2OCPActPerTensorFloat(FpOCPActPerTensorFloat, Fp8e5m2Mixin):
    """
    OCP FP8 e5m2 scaled signed act quantizer.
    """
    pass


class Fp8e5m2OCPActPerTensorFloatMSE(FpOCPActPerTensorFloatMSE, Fp8e5m2Mixin):
    """
    OCP FP8 e5m2 MSE-based scaled signed act quantizer.
    """
    pass


class Fp8e5m2OCPWeightPerTensorFloatMSE(FpOCPWeightPerTensorFloatMSE, Fp8e5m2Mixin):
    """
    OCP FP8 e5m2 MSE-based per-tensor scaled signed weight quantizer.
    """
    pass


class Fp8e5m2OCPWeightPerChannelFloatMSE(FpOCPWeightPerChannelFloatMSE, Fp8e5m2Mixin):
    """
    OCP FP8 e5m2 MSE-based per-channel scaled signed weight quantizer.
    """
    pass
