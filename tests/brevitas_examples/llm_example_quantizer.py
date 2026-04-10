from torch import nn

from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.generative.quantizers import BaseQuantizer
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY


@Registry.register(QUANTIZERS_REGISTRY, "example_int4_weight_quant")
class ExampleInt8WeightQuantizer(BaseQuantizer):
    weight_quant = Int8WeightPerTensorFloat.let(bit_width=4)


@Registry.register(QUANTIZERS_REGISTRY, "example_model_adjuster")
class ExampleModelAdjuster(BaseQuantizer):

    @classmethod
    def modify_quantized_model(cls, model: nn.Module) -> nn.Module:
        model.example_model_adjuster_applied = True
        return model


@Registry.register(QUANTIZERS_REGISTRY, "example_quant_and_model_adjuster")
class ExampleQuantAndModelAdjuster(BaseQuantizer):
    weight_quant = Int8WeightPerTensorFloat.let(bit_width=4)

    @classmethod
    def modify_quantized_model(cls, model: nn.Module) -> nn.Module:
        model.example_quant_and_model_adjuster_applied = True
        return model
