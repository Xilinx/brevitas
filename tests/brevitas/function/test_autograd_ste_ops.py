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

from typing import Tuple, Callable
from hypothesis import given
import pytest
import mock

from torch import Tensor, tensor
from brevitas.function.autograd_ste_ops import *

from tests.brevitas.common import assert_allclose
from tests.brevitas.hyp_helper import two_float_tensor_random_shape_st
from tests.brevitas.hyp_helper import scalar_float_tensor_st, scalar_float_nz_tensor_st
from tests.brevitas.function.hyp_helper import tensor_clamp_ste_min_max_scalar_tensor_test_st
from tests.brevitas.function.hyp_helper import tensor_clamp_ste_test_st
from tests.brevitas.function.hyp_helper import scalar_clamp_min_ste_test_st


# brevitas.function.autograd_ste_ops. and not brevitas.function.ops.
# in order to mock where it's used, not where it's defined
MOCK_OPS_PREFIX = 'brevitas.function.autograd_ste_ops.'


class TestElementwiseSte:

    FWD_IMPL = {
        binary_sign_ste_impl: MOCK_OPS_PREFIX + 'binary_sign',
        round_to_zero_ste_impl: MOCK_OPS_PREFIX + 'round_to_zero',
        round_ste_impl: 'torch.round',
        ceil_ste_impl: 'torch.ceil',
        floor_ste_impl: 'torch.floor',
        ternary_sign_ste_impl: 'torch.sign',
    }

    STE_IMPL = FWD_IMPL.keys()
    IDS = [fn.__qualname__ for fn in STE_IMPL]

    @given(x=two_float_tensor_random_shape_st())
    @pytest.mark.parametrize('ste_impl', STE_IMPL, ids=IDS)
    def test_fwd(self, x: Tuple[Tensor, Tensor], ste_impl: Callable):
        """
        Test that the autograd function is correctly wrapping the forward impl
        """
        fwd_impl = self.FWD_IMPL[ste_impl]
        with mock.patch(fwd_impl) as mocked_fwd_impl:
            inp, mocked_output = x
            mocked_fwd_impl.return_value = mocked_output
            output = ste_impl(inp)
            assert output is mocked_output

    @given(x=two_float_tensor_random_shape_st())
    @pytest.mark.parametrize('ste_impl', STE_IMPL, ids=IDS)
    def test_bwd(self, x: Tuple[Tensor, Tensor], ste_impl: Callable):
        """
        Test that gradients are correctly passed through
        """
        value, grad = x
        value.requires_grad_(True)
        output = ste_impl(value)
        output.backward(grad, retain_graph=True)
        assert_allclose(grad, value.grad)


class TestTensorClampSte:

    @given(x=tensor_clamp_ste_test_st())
    def test_fwd(self, x):
        """
        Test that the autograd function is correctly wrapping the forward impl
        """
        with mock.patch(MOCK_OPS_PREFIX + 'tensor_clamp') as fwd_impl:
            min_val, max_val, inp, mocked_output = x
            fwd_impl.return_value = mocked_output
            output = tensor_clamp_ste_impl(inp, min_val, max_val)
            assert output is mocked_output

    @given(x=tensor_clamp_ste_test_st())
    def test_bwd(self, x):
        """
        Test that gradients are correctly passed through to val only
        """
        min_val, max_val, val, val_grad = x
        val.requires_grad_(True)
        min_val.requires_grad_(True)
        max_val.requires_grad_(True)
        output = tensor_clamp_ste_impl(val, min_val, max_val)
        output.backward(val_grad, retain_graph=True)
        assert (min_val.grad == tensor(0.)).all()
        assert (max_val.grad == tensor(0.)).all()
        assert_allclose(val_grad, val.grad)


class TestScalarClampSte:

    @given(x=tensor_clamp_ste_test_st())
    def test_fwd(self, x):
        """
        Test that the autograd function is correctly wrapping the forward impl
        """
        with mock.patch(MOCK_OPS_PREFIX + 'tensor_clamp') as fwd_impl:
            min_val, max_val, inp, mocked_output = x
            fwd_impl.return_value = mocked_output
            output = tensor_clamp_ste_impl(inp, min_val, max_val)
            assert output is mocked_output

    @given(x=tensor_clamp_ste_min_max_scalar_tensor_test_st())
    def test_bwd(self, x):
        """
        Test that gradients are correctly passed through to val only
        """
        min_val, max_val, val, val_grad = x
        val.requires_grad_(True)
        min_val = min_val.item()
        max_val = max_val.item()
        output = scalar_clamp_ste_impl(val, min_val, max_val)
        output.backward(val_grad, retain_graph=True)
        assert_allclose(val_grad, val.grad)


class TestScalarClampMinSte:

    @given(x=tensor_clamp_ste_test_st())
    def test_fwd(self, x):
        """
        Test that the autograd function is correctly wrapping the forward impl
        """
        with mock.patch(MOCK_OPS_PREFIX + 'tensor_clamp') as fwd_impl:
            min_val, max_val, inp, mocked_output = x
            fwd_impl.return_value = mocked_output
            output = tensor_clamp_ste_impl(inp, min_val, max_val)
            assert output is mocked_output

    @given(x=scalar_clamp_min_ste_test_st())
    def test_bwd(self, x):
        """
        Test that gradients are correctly passed through to val only
        """
        min_val, val, val_grad = x
        val.requires_grad_(True)
        output = scalar_clamp_min_ste_impl(val, min_val)
        output.backward(val_grad, retain_graph=True)
        assert_allclose(val_grad, val.grad)


class TestAbsBinarySignGrad:

    @given(x=two_float_tensor_random_shape_st())
    def test_fwd(self, x):
        """
        Test that the autograd function is correctly wrapping the forward impl
        """
        with mock.patch('torch.abs') as fwd_impl:
            inp, mocked_output = x
            fwd_impl.return_value = mocked_output
            output = abs_binary_sign_grad_impl(inp)
            assert output is mocked_output

    @given(inp=scalar_float_nz_tensor_st(), grad=scalar_float_tensor_st())
    def test_bwd_nz(self, inp, grad):
        """
        Test that the backward pass matches torch.abs backward for inp != 0
        """
        import torch

        inp.requires_grad_(True)
        output = abs_binary_sign_grad_impl(inp)
        output.backward(grad)
        reference_inp = inp.detach().clone().requires_grad_(True)
        reference_output = torch.abs(reference_inp)
        reference_output.backward(grad)
        assert_allclose(inp.grad, reference_inp.grad)

    @given(grad=scalar_float_tensor_st())
    def test_bwd_zero(self, grad):
        """
        Test that the subgradient w.r.t. inp == 0 is 1 and not 0
        """
        import torch

        inp = tensor(0.0)
        inp.requires_grad_(True)
        output = abs_binary_sign_grad_impl(inp)
        output.backward(grad)
        reference_inp = inp.detach().clone().requires_grad_(True)
        reference_output = torch.abs(reference_inp)
        reference_output.backward(grad)
        assert_allclose(inp.grad, grad)
        assert reference_output == 0.0







