import pytest

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
