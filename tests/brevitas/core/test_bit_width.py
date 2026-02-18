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

    def test_bit_width_offset_clamping_min_max(self):
        """
        Test that bit_width_offset_min_val and bit_width_offset_max_val properly clamp the bit-width
        offset when values are set outside the allowed range. The min value is set to 0 for this test.
        """
        min_bit_width = 2
        bit_width_offset_min = None  # Min offset value set to 0 as requested
        bit_width_offset_max = 6  # Max offset value, allowing bit-widths up to 8
        
        # Create BitWidthParameter with min=0 and max=6
        # When min_val=None, _RestrictClampValue uses Identity() for clamping, meaning no clamping is applied
        # This test documents this behavior
        bit_width_param_no_clamp = BitWidthParameter(
            bit_width=4,
            min_bit_width=min_bit_width,
            bit_width_offset_min_val=bit_width_offset_min,
            bit_width_offset_max_val=bit_width_offset_max)
        
        # With min=0, no clamping occurs, so offset stays as-is (after abs and rounding)
        with torch.no_grad():
            bit_width_param_no_clamp.bit_width_offset.data = torch.tensor(10.0)
        
        result_no_clamp = bit_width_param_no_clamp()
        # No clamping applied, just rounded: 10 + 2 = 12
        assert_allclose(result_no_clamp, torch.tensor(12.0))
        
        # Test with non-zero min and max for actual clamping behavior
        bit_width_offset_min_nonzero = 1
        bit_width_param_clamped = BitWidthParameter(
            bit_width=4,
            min_bit_width=min_bit_width,
            bit_width_offset_min_val=bit_width_offset_min_nonzero,
            bit_width_offset_max_val=bit_width_offset_max)
        
        # Test case 1: Set offset higher than max (should clamp down to max)
        with torch.no_grad():
            bit_width_param_clamped.bit_width_offset.data = torch.tensor(10.0)
        
        result_high = bit_width_param_clamped()
        expected_max = min_bit_width + bit_width_offset_max
        assert_allclose(result_high, torch.tensor(float(expected_max)))
        
        # Test case 2: Set offset lower than min (should clamp up to min)
        with torch.no_grad():
            bit_width_param_clamped.bit_width_offset.data = torch.tensor(0.5)
        
        result_low = bit_width_param_clamped()
        expected_min = min_bit_width + bit_width_offset_min_nonzero
        assert_allclose(result_low, torch.tensor(float(expected_min)))
        
        # Test case 3: Set offset within range (should not clamp)
        with torch.no_grad():
            bit_width_param_clamped.bit_width_offset.data = torch.tensor(3.0)
        
        result_valid = bit_width_param_clamped()
        expected_valid = min_bit_width + 3.0
        assert_allclose(result_valid, torch.tensor(expected_valid))
