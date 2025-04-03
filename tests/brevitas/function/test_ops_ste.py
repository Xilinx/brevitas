# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

from hypothesis import given
import mock
import pytest
import pytest_cases
import torch

import brevitas
from brevitas import config
from brevitas.function import ops_ste
from brevitas.function.ops_ste import *
from tests.brevitas.common import BOOLS
from tests.brevitas.function.hyp_helper import scalar_clamp_min_ste_test_st
from tests.brevitas.function.hyp_helper import scalar_clamp_ste_test_st
from tests.brevitas.function.hyp_helper import tensor_clamp_ste_test_st
from tests.brevitas.hyp_helper import two_float_tensor_random_shape_st

AUTOGRAD_OPS_PREFIX = 'brevitas.ops.autograd_ste_ops.'
NATIVE_PREFIX = 'torch.ops.autograd_ste_ops.'

# name of the backend that is wrapped by each function to test
ELEMWISE_STE_BACKEND = {
    ceil_ste: 'ceil_ste_impl',
    binary_sign_ste: 'binary_sign_ste_impl',
    ternary_sign_ste: 'ternary_sign_ste_impl',
    round_ste: 'round_ste_impl',
    round_to_zero_ste: 'round_to_zero_ste_impl',
    floor_ste: 'floor_ste_impl',}


def prefix_and_status_impl(jit, native_ste):
    if native_ste and brevitas.NATIVE_STE_BACKEND_LOADED:
        if jit and config.JIT_ENABLED:
            return NATIVE_PREFIX, False
        if not jit and config.JIT_ENABLED:
            return AUTOGRAD_OPS_PREFIX, False
        if not jit and not config.JIT_ENABLED:
            return NATIVE_PREFIX, True
    if not native_ste and not brevitas.NATIVE_STE_BACKEND_LOADED:
        if jit == bool(config.JIT_ENABLED):
            return AUTOGRAD_OPS_PREFIX, True
    return None, None


def gen_case_id(jit_and_native_ste):
    """
    Generate a human readable name, so that the tests that actually run can be audited
    """
    jit, native_ste = jit_and_native_ste
    prefix, status = prefix_and_status_impl(jit, native_ste)
    if prefix is None:
        return f"jit={int(jit)}-native_ste={int(native_ste)}"
    else:
        return f"prefix={prefix.split('.')[0]}-called={int(status)}-jit={int(bool(config.JIT_ENABLED))}-native_ste={int(native_ste)}"


@pytest_cases.fixture()
@pytest_cases.parametrize(
    "jit, native_ste",
    [(j, s) for j in BOOLS for s in BOOLS],
    ids=gen_case_id,
)
def prefix_and_status(jit, native_ste) -> Tuple[str, bool]:
    """
    Fixture for prefixes and expected result of downstream tests of ste backends.

    In general, the tests contained in this file check some brevitas configuration and check that the correct brevitas functions are called.
    However, when `config.JIT_ENABLED=True` and `config.NATIVE_STE_BACKEND_LOADED=True`,
    we won't see that the correct function is called, because the full compute graph is compiled down to c++.
    In this case, we check return `status=False` and check that neither prefix is called.
    """
    prefix, status = prefix_and_status_impl(jit, native_ste)
    if prefix is None:
        pytest.skip()
    else:
        return prefix, status


def test_jit_annotations(prefix_and_status: Tuple[str, bool]):
    """
    Test that the annotations to enable/disable the jit are being set correctly
    """
    prefix, status = prefix_and_status
    if brevitas.NATIVE_STE_BACKEND_LOADED:
        assert prefix == NATIVE_PREFIX or prefix == AUTOGRAD_OPS_PREFIX  # Sanity-check, should always be True
        assert ops_ste.fn_prefix == torch
        assert ops_ste.script_flag == brevitas.jit.script
    else:
        assert prefix == AUTOGRAD_OPS_PREFIX
        assert ops_ste.fn_prefix == brevitas
        assert ops_ste.script_flag == torch.jit.ignore
    if config.JIT_ENABLED:
        assert brevitas.jit.script == torch.jit.script
    else:
        assert brevitas.jit.script == brevitas.jit._disabled


@given(x=two_float_tensor_random_shape_st())
@pytest.mark.parametrize('ste_impl', ELEMWISE_STE_BACKEND.keys())
def test_elemwise_ste_backend(prefix_and_status: Tuple[str, bool], x, ste_impl):
    """
    Test that ste_impl is wrapping the corresponding backend implementation correctly.
    """
    prefix, status = prefix_and_status
    backend_name = ELEMWISE_STE_BACKEND[ste_impl]
    with mock.patch(prefix + backend_name) as python_backend:
        inp, mocked_return_val = x
        python_backend.return_value = mocked_return_val
        return_val = ste_impl(inp)
        if status:
            # check that the wrapped function is called with the correct argument
            python_backend.assert_called_once_with(inp)
            # check that the return value of the wrapper is the return values of the wrapped function
            assert return_val is mocked_return_val
        else:
            # If (config.JIT_ENABLED and brevitas.NATIVE_STE_BACKEND_LOADED) we expect the prefix won't be called
            python_backend.assert_not_called()


@given(x=tensor_clamp_ste_test_st())
def test_tensor_clamp_ste_backend(prefix_and_status: Tuple[str, bool], x):
    """
    Test that tensor_clamp_ste is wrapping the backend implementation correctly.
    """
    prefix, status = prefix_and_status
    backend_name = 'tensor_clamp_ste_impl'
    with mock.patch(prefix + backend_name) as python_backend:
        min_val, max_val, inp, mocked_return_val = x
        python_backend.return_value = mocked_return_val
        return_val = tensor_clamp_ste(inp, min_val, max_val)
        if status:
            # check that the wrapped function is called with the correct argument
            python_backend.assert_called_once_with(inp, min_val, max_val)
            # check that the return value of the wrapper is the return values of the wrapped function
            assert return_val is mocked_return_val
        else:
            # If (config.JIT_ENABLED and brevitas.NATIVE_STE_BACKEND_LOADED) we expect the prefix won't be called
            python_backend.assert_not_called()


@given(x=scalar_clamp_ste_test_st())
def test_scalar_clamp_ste_backend(prefix_and_status: Tuple[str, bool], x):
    """
    Test that scalar_clamp_ste is wrapping the backend implementation correctly.
    """
    prefix, status = prefix_and_status
    backend_name = 'scalar_clamp_ste_impl'
    with mock.patch(prefix + backend_name) as python_backend:
        min_val, max_val, inp, mocked_return_val = x
        python_backend.return_value = mocked_return_val
        return_val = scalar_clamp_ste(inp, min_val, max_val)
        if status:
            # check that the wrapped function is called with the correct argument
            python_backend.assert_called_once_with(inp, min_val, max_val)
            # check that the return value of the wrapper is the return values of the wrapped function
            assert return_val is mocked_return_val
        else:
            # If (config.JIT_ENABLED and brevitas.NATIVE_STE_BACKEND_LOADED) we expect the prefix won't be called
            python_backend.assert_not_called()


@given(x=scalar_clamp_min_ste_test_st())
def test_scalar_clamp_min_ste_backend(prefix_and_status: Tuple[str, bool], x):
    """
    Test that scalar_clamp_min_ste is wrapping the backend implementation correctly.
    """
    prefix, status = prefix_and_status
    backend_name = 'scalar_clamp_min_ste_impl'
    with mock.patch(prefix + backend_name) as python_backend:
        min_val, inp, mocked_return_val = x
        python_backend.return_value = mocked_return_val
        return_val = scalar_clamp_min_ste(inp, min_val)
        if status:
            # check that the wrapped function is called with the correct argument
            python_backend.assert_called_once_with(inp, min_val)
            # check that the return value of the wrapper is the return values of the wrapped function
            assert return_val is mocked_return_val
        else:
            # If (config.JIT_ENABLED and brevitas.NATIVE_STE_BACKEND_LOADED) we expect the prefix won't be called
            python_backend.assert_not_called()
