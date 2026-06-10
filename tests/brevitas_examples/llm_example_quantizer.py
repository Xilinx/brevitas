from torch import nn

from brevitas.quant.scaled_int import Int8ActPerTensorFloat
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
    def post_process_quant_model(cls, model: nn.Module) -> nn.Module:
        model.example_model_adjuster_applied = True
        return model


@Registry.register(QUANTIZERS_REGISTRY, "example_quant_and_model_adjuster")
class ExampleQuantAndModelAdjuster(BaseQuantizer):
    weight_quant = Int8WeightPerTensorFloat.let(bit_width=4)
    linear_input_quant = Int8ActPerTensorFloat

    @classmethod
    def post_process_quant_model(cls, model: nn.Module) -> nn.Module:
        model.example_quant_and_model_adjuster_applied = True
        for m in model.model.layers:
            # Tie input_quant
            base_quant_qkv = m.self_attn.q_proj.input_quant
            m.self_attn.v_proj.input_quant = base_quant_qkv
            m.self_attn.k_proj.input_quant = base_quant_qkv

            base_quant_up_gate = m.mlp.gate_proj.input_quant
            m.mlp.up_proj.input_quant = base_quant_up_gate

            # Tie weight_quant
            base_quant_qkv = m.self_attn.q_proj.weight_quant
            m.self_attn.v_proj.weight_quant = base_quant_qkv
            m.self_attn.k_proj.weight_quant = base_quant_qkv

            base_quant_up_gate = m.mlp.gate_proj.weight_quant
            m.mlp.up_proj.weight_quant = base_quant_up_gate
        return model
