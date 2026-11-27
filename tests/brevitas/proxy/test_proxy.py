# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gc
import weakref

import pytest
import torch

from brevitas.core.scaling import ParameterFromStatsFromParameterScaling
from brevitas.nn import QuantLinear
from brevitas.nn.quant_activation import QuantReLU
from brevitas.quant.scaled_int import Int8AccumulatorAwareWeightQuant
from brevitas.quant.scaled_int import Int8BiasPerTensorFloatInternalScaling
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatDecoupled
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerTensorFloat
from tests.marker import jit_disabled_for_dynamic_quant_act


class TestProxy:

    def test_bias_proxy(self):
        model = QuantLinear(10, 5, bias_quant=Int8BiasPerTensorFloatInternalScaling)
        assert model.bias_quant.scale() is not None
        assert model.bias_quant.zero_point() is not None
        assert model.bias_quant.bit_width() is not None

        model.bias_quant.disable_quant = True
        assert model.bias_quant.scale() is None
        assert model.bias_quant.zero_point() is None
        assert model.bias_quant.bit_width() is None

    def test_weight_proxy(self):
        model = QuantLinear(10, 5, weight_quant=Int8WeightPerTensorFloat)
        assert model.weight_quant.scale() is not None
        assert model.weight_quant.zero_point() is not None
        assert model.weight_quant.bit_width() is not None

        model.weight_quant.disable_quant = True
        assert model.weight_quant.scale() is None
        assert model.weight_quant.zero_point() is None
        assert model.weight_quant.bit_width() is None

    def test_tracked_parameter_list_follows_parameter_replacement(self):
        model = QuantLinear(
            10,
            5,
            weight_quant=Int8WeightPerTensorFloat,
            weight_scaling_impl_type='parameter_from_stats')
        assert isinstance(
            model.weight_quant.tensor_quant.scaling_impl, ParameterFromStatsFromParameterScaling)
        tracked_parameters = model.weight_quant.quant_injector.tracked_parameter_list
        old_weight = model.weight
        old_weight_ref = weakref.ref(old_weight)
        new_weight = torch.nn.Parameter(torch.randn_like(old_weight))

        # FSDP2 swaps parameters directly in the module parameter dictionary.
        model._parameters['weight'] = new_weight
        del old_weight
        gc.collect()

        assert tracked_parameters[0] is new_weight
        assert old_weight_ref() is None

    def test_weight_decoupled_proxy(self):
        model = QuantLinear(10, 5, weight_quant=Int8WeightPerChannelFloatDecoupled)
        assert model.weight_quant.pre_scale() is not None
        assert model.weight_quant.pre_zero_point() is not None

        model.weight_quant.disable_quant = True
        assert model.weight_quant.pre_scale() is None
        assert model.weight_quant.pre_zero_point() is None

    def test_weight_decoupled_with_input_proxy(self):
        model = QuantLinear(10, 5, weight_quant=Int8AccumulatorAwareWeightQuant)
        with pytest.raises(AssertionError):
            model.weight_quant.scale()
        with pytest.raises(AssertionError):
            model.weight_quant.zero_point()

        with pytest.raises(NotImplementedError):
            model.weight_quant.pre_scale()
        with pytest.raises(NotImplementedError):
            model.weight_quant.pre_zero_point()

    def test_act_proxy(self):
        model = QuantReLU()
        assert model.act_quant.scale() is not None
        assert model.act_quant.zero_point() is not None
        assert model.act_quant.bit_width() is not None

        model.act_quant.disable_quant = True
        assert model.act_quant.scale() is None
        assert model.act_quant.zero_point() is None
        assert model.act_quant.bit_width() is None

    @jit_disabled_for_dynamic_quant_act()
    def test_dynamic_act_proxy(self):
        model = QuantReLU(Int8DynamicActPerTensorFloat)

        with pytest.raises(RuntimeError, match="Scale for Dynamic Act Quant is input-dependant"):
            model.act_quant.scale()
        with pytest.raises(RuntimeError,
                           match="Zero point for Dynamic Act Quant is input-dependant"):
            model.act_quant.zero_point()

        assert model.act_quant.bit_width() is not None

        model.act_quant.disable_quant = True
        assert model.act_quant.bit_width() is None

    def test_training_state(self):
        quant_layer = QuantLinear(10, 5, weight_quant=Int8WeightPerTensorFloat)
        quant_layer.eval()

        # Setting new weights will re-init the quant tensor
        quant_layer.weight = torch.nn.Parameter(torch.randn_like(quant_layer.weight))

        assert quant_layer.weight_quant.tensor_quant.training == False
