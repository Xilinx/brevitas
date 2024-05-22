# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from hypothesis import given
import pytest
import pytest_cases
import torch

from brevitas import config
from brevitas.core.bit_width import BitWidthParameter
from brevitas.core.bit_width import BitWidthStatefulConst
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.restrict_val import IntRestrictValue
from tests.brevitas.common import assert_allclose
from tests.brevitas.core.bit_width_fixture import *  # noqa
from tests.brevitas.hyp_helper import scalar_float_nz_tensor_st
from tests.marker import skip_on_macos_nox


class TestBitWidthAll:
    """
    Test invariants on all bit-widths variants
    """

    def test_return_value(self, bit_width_all, bit_width_init, bit_width_init_two):
        bit_width_tensor = bit_width_all()
        # BitWidthStatefulConst is initialized from bit_width_init_two
        if isinstance(bit_width_all, BitWidthStatefulConst):
            bit_width_init = bit_width_init_two
        assert bit_width_tensor == bit_width_init

    def test_return_datatype(self, bit_width_all):
        bit_width_tensor = bit_width_all()
        assert bit_width_tensor.dtype == torch.float32


class TestBitWidthParameterDefaults:
    """
    Test default values of BitWidthParameter
    """

    def test_default_restrict_bit_width_impl(self, bit_width_parameter_defaults):
        bit_width_module = bit_width_parameter_defaults
        assert isinstance(bit_width_module.restrict_bit_width_impl, IntRestrictValue)

    def test_default_float_to_int_impl(self, bit_width_parameter_defaults):
        bit_width_module = bit_width_parameter_defaults
        assert isinstance(bit_width_module.restrict_bit_width_impl.float_to_int_impl, RoundSte)

    def test_bit_width_base(self, bit_width_parameter_defaults):
        bit_width_module = bit_width_parameter_defaults
        assert bit_width_module.bit_width_base == 2

    def test_bit_width_offset(self, bit_width_parameter_defaults, bit_width_init):
        bit_width_module = bit_width_parameter_defaults
        assert bit_width_module.bit_width_offset == bit_width_init - 2

    def test_override_pretrained(self, bit_width_parameter_defaults):
        bit_width_module = bit_width_parameter_defaults
        assert bit_width_module.override_pretrained == False


class TestBitWidthParameter:

    @pytest.mark.xfail(raises=RuntimeError, strict=True)
    def test_init_fail(self, bit_width_init, min_bit_width_init):
        """
        Test that BitWidthParameter init fails when bit_width is less than min_bit_width
        """
        if bit_width_init < min_bit_width_init:
            BitWidthParameter(bit_width_init, min_bit_width=min_bit_width_init)
        else:
            pytest.skip('Skip expected legal cases')

    def clean_up_bwd(self, bit_width_parameter: BitWidthParameter):
        """
        Manually clean up after fixtures w/ function scope + hypothesis tests
        https://github.com/HypothesisWorks/hypothesis/issues/377
        https://github.com/pytest-dev/pytest/issues/916
        """
        bit_width_parameter.bit_width_offset.grad = None

    @given(bit_width_grad=scalar_float_nz_tensor_st())
    def test_bwd(self, bit_width_parameter: BitWidthParameter, bit_width_grad):
        """
        Test that gradients are propagated to bit_width_parameter.bit_width_offset
        """
        bit_width_tensor = bit_width_parameter()
        bit_width_tensor.backward(bit_width_grad)
        assert_allclose(bit_width_parameter.bit_width_offset.grad, bit_width_grad)
        self.clean_up_bwd(bit_width_parameter)

    def test_bit_width_base(self, bit_width_parameter, min_bit_width_init):
        assert bit_width_parameter.bit_width_base == min_bit_width_init

    def test_bit_width_offset(
            self, bit_width_parameter: BitWidthParameter, bit_width_init, min_bit_width_init):
        """
        Test that bit_width_offset is initialized corrected
        """
        assert bit_width_init >= min_bit_width_init
        assert bit_width_parameter.bit_width_offset == bit_width_init - min_bit_width_init

    def test_override_pretrained(self, bit_width_parameter, override_pretrained):
        assert bit_width_parameter.override_pretrained == override_pretrained

    @pytest_cases.parametrize(
        'ignore_missing_keys',
        [True, pytest.param(False, marks=pytest.mark.xfail(raises=RuntimeError))])
    def test_ignore_missing_keys(self, bit_width_stateful, ignore_missing_keys):
        """
        Test that config.IGNORE_MISSING_KEYS is read correctly
        """
        config.IGNORE_MISSING_KEYS = ignore_missing_keys
        bit_width_stateful.load_state_dict({})

    def test_override_pretrained_value(self, bit_width_parameter, override_pretrained):
        """
        Test that override_pretrained is read correctly
        """
        override_value = bit_width_parameter.bit_width_offset
        state_dict_value = bit_width_parameter.bit_width_offset + 1
        config.IGNORE_MISSING_KEYS = True  # always ignore missing keys
        bit_width_parameter.load_state_dict({'bit_width_offset': torch.tensor(state_dict_value)})
        value = bit_width_parameter.bit_width_offset
        if override_pretrained:
            assert value == override_value
        else:
            assert value == state_dict_value

    @skip_on_macos_nox
    def test_load_from_stateful_const(
            self,
            bit_width_parameter,
            bit_width_stateful_const,
            bit_width_init,
            min_bit_width_init,
            bit_width_init_two,
            override_pretrained):
        """
        Test state dictionary from BitWidthStatefulConst is read correctly
        """
        if (bit_width_init_two < min_bit_width_init) and not override_pretrained:
            pytest.xfail('bit_width cannot be smaller than min_bit_width')

        override_value = bit_width_parameter.bit_width_offset
        bit_width_parameter.load_state_dict(bit_width_stateful_const.state_dict())
        bit_width_parameter_tensor = bit_width_parameter()
        if override_pretrained:
            assert bit_width_parameter.bit_width_offset == override_value
            assert bit_width_parameter_tensor == bit_width_init
        else:
            bit_width_stateful_const_tensor = bit_width_stateful_const()
            assert bit_width_stateful_const_tensor == bit_width_init_two
            assert bit_width_stateful_const_tensor == bit_width_parameter_tensor
