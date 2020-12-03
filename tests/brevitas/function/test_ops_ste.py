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

import pytest
import mock
from hypothesis import given

import torch

import brevitas
from brevitas import config
from brevitas.function import ops_ste, autograd_ste_ops
from brevitas.function.ops_ste import *

from tests.brevitas.hyp_helper import two_float_tensor_random_shape_st
from tests.brevitas.function.hyp_helper import tensor_clamp_ste_test_st
from tests.brevitas.function.hyp_helper import scalar_clamp_min_ste_test_st


AUTOGRAD_OPS_PREFIX = 'brevitas.function.autograd_ste_ops.'
NATIVE_PREFIX = 'torch.ops.autograd_ste_ops.'

# name of the backend that is wrapped by each function to test
ELEMWISE_STE_BACKEND = {
    ceil_ste: 'ceil_ste_impl',
    binary_sign_ste: 'binary_sign_ste_impl',
    ternary_sign_ste: 'ternary_sign_ste_impl',
    round_ste: 'round_ste_impl',
    round_to_zero_ste: 'round_to_zero_ste_impl',
    floor_ste: 'floor_ste_impl',
}


@pytest.fixture()
def prefix() -> str:
    """
    Fixture for prefix of the ste backend
    """
    if brevitas.NATIVE_STE_BACKEND_LOADED:
        prefix = NATIVE_PREFIX
    else:
        prefix = AUTOGRAD_OPS_PREFIX
    return prefix


def test_jit_annotations(prefix: str):
    """
    Test that the annotations to enable/disable the jit are being set correctly
    """
    if prefix == NATIVE_PREFIX:
        assert ops_ste.fn_prefix == torch.ops.autograd_ste_ops
        assert ops_ste.script_flag == brevitas.jit.script
        assert config.JIT_ENABLED
    else:
        assert prefix == AUTOGRAD_OPS_PREFIX
        assert ops_ste.fn_prefix == autograd_ste_ops
        assert ops_ste.script_flag == torch.jit.ignore


@given(x=two_float_tensor_random_shape_st())
@pytest.mark.parametrize('ste_impl', ELEMWISE_STE_BACKEND.keys())
def test_elemwise_ste_backend(prefix, x, ste_impl):
    """
    Test that ste_impl is wrapping the corresponding backend implementation correctly.
    """
    backend_name = ELEMWISE_STE_BACKEND[ste_impl]
    with mock.patch(prefix + backend_name) as python_backend:
        inp, mocked_return_val = x
        python_backend.return_value = mocked_return_val
        return_val = ste_impl(inp)
        # check that the wrapped function is called with the correct argument
        python_backend.assert_called_once_with(inp)
        # check that the return value of the wrapper is the return values of the wrapped function
        assert return_val is mocked_return_val


@given(x=tensor_clamp_ste_test_st())
def test_tensor_clamp_ste_backend(prefix: str, x):
    """
    Test that tensor_clamp_ste is wrapping the backend implementation correctly.
    """
    backend_name = 'tensor_clamp_ste_impl'
    with mock.patch(prefix + backend_name) as python_backend:
        min_val, max_val, inp, mocked_return_val = x
        python_backend.return_value = mocked_return_val
        return_val = tensor_clamp_ste(inp, min_val, max_val)
        # check that the wrapped function is called with the correct argument
        python_backend.assert_called_once_with(inp, min_val, max_val)
        # check that the return value of the wrapper is the return values of the wrapped function
        assert return_val is mocked_return_val


@given(x=tensor_clamp_ste_test_st())
def test_scalar_clamp_ste_backend(prefix: str, x):
    """
    Test that scalar_clamp_ste is wrapping the backend implementation correctly.
    """
    backend_name = 'scalar_clamp_ste_impl'
    with mock.patch(prefix + backend_name) as python_backend:
        min_val, max_val, inp, mocked_return_val = x
        python_backend.return_value = mocked_return_val
        return_val = scalar_clamp_ste(inp, min_val, max_val)
        # check that the wrapped function is called with the correct argument
        python_backend.assert_called_once_with(inp, min_val, max_val)
        # check that the return value of the wrapper is the return values of the wrapped function
        assert return_val is mocked_return_val


@given(x=scalar_clamp_min_ste_test_st())
def test_scalar_clamp_min_ste_backend(prefix: str, x):
    """
    Test that scalar_clamp_min_ste is wrapping the backend implementation correctly.
    """
    backend_name = 'scalar_clamp_min_ste_impl'
    with mock.patch(prefix + backend_name) as python_backend:
        inp, min_val, mocked_return_val = x
        python_backend.return_value = mocked_return_val
        return_val = scalar_clamp_min_ste(inp, min_val)
        # check that the wrapped function is called with the correct argument
        python_backend.assert_called_once_with(inp, min_val)
        # check that the return value of the wrapper is the return values of the wrapped function
        assert return_val is mocked_return_val