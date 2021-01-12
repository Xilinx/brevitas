# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import mock
from hypothesis import given

import torch
from brevitas.core.quant import *
from brevitas.core.function_wrapper import RoundSte, TensorClamp

from tests.brevitas.hyp_helper import float_tensor_random_shape_st, scalar_float_p_tensor_st
from tests.brevitas.core.shared_quant_fixture import * # noqa
from tests.brevitas.core.int_quant_fixture import * # noqa
from tests.brevitas.core.bit_width_fixture import * # noqa


class TestIntQuantUnit:

    @given(
        inp=float_tensor_random_shape_st(),
        scale=scalar_float_p_tensor_st(),
        zero_point=scalar_float_p_tensor_st())
    def test_int_quant_to_int_called_with(
            self, inp, narrow_range, signed, bit_width_init, zero_point, scale):
        float_to_int_impl = mock.Mock(side_effect=lambda x: x)
        tensor_clamp_impl = mock.Mock(side_effect=lambda x, min_val, max_val: x)
        int_quant = IntQuant(
            narrow_range=narrow_range,
            signed=signed,
            float_to_int_impl=float_to_int_impl,
            tensor_clamp_impl=tensor_clamp_impl)
        bit_width = torch.tensor(bit_width_init)
        output = int_quant.to_int(scale, zero_point, bit_width, inp)
        float_to_int_impl.assert_called_once_with(output)
        tensor_clamp_impl.assert_called_once_with(
            output, min_val=int_quant.min_int(bit_width), max_val=int_quant.max_int(bit_width))

    def test_int_quant_defaults(self, narrow_range, signed):
        int_quant = IntQuant(narrow_range=narrow_range, signed=signed)
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
        int_quant = IntQuant(narrow_range=narrow_range, signed=signed)
        zero_point = torch.tensor(zero_point_init).float()
        bit_width = torch.tensor(bit_width_init).float()
        scale = torch.tensor(standalone_scaling_init).float()
        # apply scale and zero point to the input distribution
        inp = scale * (arange_int_tensor - zero_point).float()
        output = int_quant(scale, zero_point, bit_width, inp)
        assert torch.isclose(inp, output).all()
