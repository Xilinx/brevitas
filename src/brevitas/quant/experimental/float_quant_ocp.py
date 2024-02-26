# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.experimental.float_base import FloatActBase
from brevitas.quant.experimental.float_base import FloatWeightBase
from brevitas.quant.experimental.float_base import Fp8e4m3Mixin
from brevitas.quant.experimental.float_base import Fp8e5m2Mixin
from brevitas.quant.experimental.float_base import ScaledFloatActBase
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase


class Fp8e4m3OCPMixin(Fp8e4m3Mixin):
    nan_values = (('111',))
    inf_values = None


class Fp8e5m2OCPMixin(Fp8e5m2Mixin):
    nan_values = ('01', '11', '10')
    inf_values = (('00',))


class Fp8e4m3OCPWeight(Fp8e4m3OCPMixin, FloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer.
    """
    pass


class Fp8e5m2OCPWeight(Fp8e5m2OCPMixin, FloatWeightBase):
    """
    FP8 signed E5M2 weight quantizer.
    """
    pass


class Fp8e4m3OCPAct(Fp8e4m3OCPMixin, FloatActBase):
    """
    FP8 signed E4M3 activation quantizer.
    """
    pass


class Fp8e5m2OCPAct(Fp8e5m2OCPMixin, FloatActBase):
    """
    FP8 signed E5M2 activation quantizer.
    """
    pass


class Fp8e4m3OCPWeightPerTensorFloat(Fp8e4m3OCPMixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e5m2OCPWeightPerTensorFloat(Fp8e5m2OCPMixin, ScaledFloatWeightBase):
    """
    FP8 signed E5M2 weight quantizer with per-tensor absmax-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e4m3OCPActPerTensorFloat(Fp8e4m3OCPMixin, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e5m2OCPActPerTensorFloat(Fp8e5m2OCPMixin, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-tensor static percentile-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e4m3OCPWeightPerChannelFloat(Fp8e4m3OCPMixin, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_channel = True


class Fp8e5m2OCPWeightPerChannelFloat(Fp8e5m2OCPMixin, ScaledFloatWeightBase):
    """
    FP8 signed E5M2 weight quantizer with per-channel absmax-based scaling.
    """
    scaling_per_output_channel = True


class Fp8e4m3OCPActPerChannelFloat2d(Fp8e4m3OCPMixin, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e5m2OCPActPerChannelFloat2d(Fp8e5m2OCPMixin, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-channel static percentile-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e4m3OCPActPerTensorFloatMSE(Fp8e4m3OCPMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e5m2OCPActPerTensorFloatMSE(Fp8e5m2OCPMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-tensor static MSE-based scaling.
    """
    scaling_per_output_channel = False


class Fp8e4m3OCPActPerChannelFloat2dMSE(Fp8e4m3OCPMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E4M3 activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e5m2OCPActPerChannelFloat2dMSE(Fp8e5m2OCPMixin, MSESymmetricScale, ScaledFloatActBase):
    """
    FP8 signed E5M2 activation quantizer with per-channel static MSE-based scaling.
    """
    scaling_per_output_channel = True
    scaling_stats_permute_dims = (1, 0, 2, 3)


class Fp8e4m3OCPWeightPerChannelFloatMSE(Fp8e4m3OCPMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-channel MSE-based scaling.
    """
    scaling_per_output_channel = True


class Fp8e4m3OCPWeightPerTensorFloatMSE(Fp8e4m3OCPMixin, MSESymmetricScale, ScaledFloatWeightBase):
    """
    FP8 signed E3M4 weight quantizer with per-tensor MSE-based scaling.
    """
    scaling_per_output_channel = False
