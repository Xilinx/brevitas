from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.generative.quantizers import BaseQuantizer
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY


@Registry.register(QUANTIZERS_REGISTRY, "example_int4_weight_quant")
class ExampleInt8WeightQuantizer(BaseQuantizer):
    weight_quant = Int8WeightPerTensorFloat.let(bit_width=4)
