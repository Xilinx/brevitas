# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from _dependencies.exceptions import DependencyError
import pytest

from brevitas.inject import ExtendedInjector
from brevitas.nn import QuantLinear
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat


class ResolvableAttributeInjector(ExtendedInjector):
    known = 1


def test_missing_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        ResolvableAttributeInjector.missing


def test_missing_attribute_keeps_getattr_default():
    assert getattr(ResolvableAttributeInjector, 'missing', None) is None
    assert not hasattr(ResolvableAttributeInjector, 'missing')


def test_missing_attribute_is_still_a_dependency_error():
    with pytest.raises(DependencyError):
        ResolvableAttributeInjector.missing


def test_accelerate_hook_removal_probes_quantizer_attributes():
    hooks = pytest.importorskip('accelerate.hooks')
    quant_linear = QuantLinear(
        16,
        16,
        bias=False,
        weight_quant=Int8WeightPerTensorFloat,
        input_quant=Int8ActPerTensorFloat)

    # accelerate >= 1.16 probes every value stored in a module __dict__ with a getattr default
    # while removing hooks, so a missing attribute has to raise AttributeError.
    quant_injector = quant_linear.weight_quant.quant_injector
    assert getattr(quant_injector, '_accelerate_added_attribute', False) is False

    hooks.remove_hook_from_module(quant_linear, recurse=True)
