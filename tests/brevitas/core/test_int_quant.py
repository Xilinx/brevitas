# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import mock
import torch

from brevitas.core.function_wrapper import Identity
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant import *
from tests.brevitas.core.bit_width_fixture import *  # noqa
from tests.brevitas.core.int_quant_fixture import *  # noqa
from tests.brevitas.core.shared_quant_fixture import *  # noqa
from tests.brevitas.hyp_helper import float_tensor_random_shape_st
from tests.brevitas.hyp_helper import scalar_float_p_tensor_st
from tests.marker import jit_disabled_for_mock


class TestIntQuantUnit:

    @given(
        inp=float_tensor_random_shape_st(),
        scale=scalar_float_p_tensor_st(),
        zero_point=scalar_float_p_tensor_st())
    @jit_disabled_for_mock()
    def test_int_quant_to_int_called_with(
            self, inp, narrow_range, signed, bit_width_init, zero_point, scale):
        float_to_int_impl = mock.Mock(side_effect=lambda x: x)
        tensor_clamp_impl = mock.Mock(side_effect=lambda x, min_val, max_val: x)
        int_quant = IntQuant(
            narrow_range=narrow_range,
            signed=signed,
            input_view_impl=Identity(),
            float_to_int_impl=float_to_int_impl,
            tensor_clamp_impl=tensor_clamp_impl)
        bit_width = torch.tensor(bit_width_init)
        output = int_quant.to_int(scale, zero_point, bit_width, inp)
        float_to_int_impl.assert_called_once_with(output)
        tensor_clamp_impl.assert_called_once_with(
            output, min_val=int_quant.min_int(bit_width), max_val=int_quant.max_int(bit_width))

    def test_int_quant_defaults(self, narrow_range, signed):
        int_quant = IntQuant(narrow_range=narrow_range, signed=signed, input_view_impl=Identity())
        assert isinstance(int_quant.float_to_int_impl, RoundSte)
        assert isinstance(int_quant.tensor_clamp_impl, TensorClamp)

    def test_int_quant_arange(
            self,
            narrow_range,
            signed,
            standalone_scaling_init,
            zero_point_init,
            bit_width_init,
            arange_int_tensor):
        int_quant = IntQuant(narrow_range=narrow_range, signed=signed, input_view_impl=Identity())
        zero_point = torch.tensor(zero_point_init).float()
        bit_width = torch.tensor(bit_width_init).float()
        scale = torch.tensor(standalone_scaling_init).float()
        # apply scale and zero point to the input distribution
        inp = scale * (arange_int_tensor - zero_point).float()
        output = int_quant(scale, zero_point, bit_width, inp)
        assert torch.isclose(inp, output).all()
