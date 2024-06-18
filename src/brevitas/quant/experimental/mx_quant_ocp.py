# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from brevitas.core.function_wrapper.ops_ste import CeilSte
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import RestrictValueType
from brevitas.quant.experimental.float_base import ScaledFloatWeightBase
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloat
from brevitas.quant.experimental.float_quant_ocp import Fp8e5m2OCPWeightPerTensorFloat


class Fp6e3m2OCPMixin(ExtendedInjector):
    bit_width = 6
    exponent_bit_width = 3
    mantissa_bit_width = 2
    nan_values = None
    inf_values = None


class Fp6e2m3OCPMixin(ExtendedInjector):
    bit_width = 6
    exponent_bit_width = 2
    mantissa_bit_width = 3
    nan_values = None
    inf_values = None


class Fp4e2m1OCPMixin(ExtendedInjector):
    bit_width = 4
    exponent_bit_width = 2
    mantissa_bit_width = 1
    nan_values = None
    inf_values = None


class MXWeightMixIn(ExtendedInjector):
    group_size = 32
    restrict_scaling_type = RestrictValueType.POWER_OF_TWO
    restrict_value_float_to_int_impl = CeilSte


class MXFp8e4m3OCPWeightPerTensorFloat(Fp8e4m3OCPWeightPerTensorFloat, MXWeightMixIn):
    pass


class MXFp8e5m2OCPWeightPerTensorFloat(Fp8e5m2OCPWeightPerTensorFloat, MXWeightMixIn):
    pass


class MXFp6e3m2OCPWeightPerTensorFloat(Fp6e3m2OCPMixin, ScaledFloatWeightBase, MXWeightMixIn):
    pass


class MXFp6e2m3OCPWeightPerTensorFloat(Fp6e2m3OCPMixin, ScaledFloatWeightBase, MXWeightMixIn):
    pass


class MXFp4e2m1OCPWeightPerTensorFloat(Fp4e2m1OCPMixin, ScaledFloatWeightBase, MXWeightMixIn):
    pass
