# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import mock
import torch

from brevitas.core.function_wrapper import RoundSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant import TruncIntQuant
from brevitas.core.scaling import TruncMsbScaling
from tests.brevitas.core.bit_width_fixture import *  # noqa
from tests.brevitas.core.int_quant_fixture import *  # noqa


class TestTruncIntQuantUnit:

    def test_trunc_int_quant_defaults(self, bit_width_const):
        trunc_int_quant = TruncIntQuant(
            bit_width_impl=bit_width_const, float_to_int_impl=RoundSte())
        assert isinstance(trunc_int_quant.tensor_clamp_impl, TensorClamp)
        assert isinstance(trunc_int_quant.trunc_scaling_impl, TruncMsbScaling)
        assert trunc_int_quant.narrow_range == False
