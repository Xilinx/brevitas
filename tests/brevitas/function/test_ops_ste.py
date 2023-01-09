# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import mock
from hypothesis import given

import torch

import brevitas
from brevitas import config
from brevitas.function import ops_ste
from brevitas.function.ops_ste import *

from tests.brevitas.hyp_helper import two_float_tensor_random_shape_st
from tests.brevitas.function.hyp_helper import tensor_clamp_ste_test_st
from tests.brevitas.function.hyp_helper import scalar_clamp_min_ste_test_st


AUTOGRAD_OPS_PREFIX = 'brevitas.ops.autograd_ste_ops.'
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
        assert ops_ste.fn_prefix == torch
        assert ops_ste.script_flag == brevitas.jit.script
        assert config.JIT_ENABLED
    else:
        assert prefix == AUTOGRAD_OPS_PREFIX
        assert ops_ste.fn_prefix == brevitas
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