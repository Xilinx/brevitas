from dataclasses import dataclass
from dataclasses import field

from torch import nn

from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.generative.quantizers import BaseQuantizer
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import GeneralizedTrainer
from brevitas_examples.llm.llm_quant.rotation_optimization import OptimizerConfig
from brevitas_examples.llm.llm_quant.rotation_optimization import TRAINER_REGISTRY
from brevitas_examples.llm.llm_quant.rotation_optimization import TrainingArguments


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


@dataclass
class ExampleTrainingArguments(TrainingArguments):
    pass


class ExampleTrainer(GeneralizedTrainer):
    pass


class RegisteredExampleTrainingArguments(ExampleTrainingArguments):
    pass


class RegisteredExampleTrainer(ExampleTrainer):
    training_args_cls = RegisteredExampleTrainingArguments


TRAINER_REGISTRY.register("minimal_trainer")(RegisteredExampleTrainer)


# ---------------------------------------------------------------------------
# Example trainer with a single optimizer handling two parameter groups.
#
# ``OptimizerConfig.params`` is a list of two selection callables (one per
# parameter group), and the matching ``optimizer_scheduler_args`` entry provides
# one optimizer with a per-group ``optimizer_kwargs`` list of the same length.
# ---------------------------------------------------------------------------
def _select_q_proj_params(model, training_args):
    return [p for name, p in model.named_parameters() if "q_proj" in name]


def _select_non_q_proj_params(model, training_args):
    return [p for name, p in model.named_parameters() if "q_proj" not in name]


def _build_two_group_optimizer_setup():
    # A single OptimizerConfig with two parameter-selection callables -> a single
    # optimizer with two parameter groups: the q_proj parameters and everything
    # else.
    return [OptimizerConfig(params=[_select_q_proj_params, _select_non_q_proj_params])]


@dataclass
class TwoGroupExampleTrainingArguments(TrainingArguments):
    # Learning rates for the two parameter groups of the single optimizer.
    q_proj_lr: float = field(default=1e-3, metadata={"help": "LR for the q_proj parameter group."})
    non_q_proj_lr: float = field(
        default=1e-2, metadata={"help": "LR for the non-q_proj parameter group."})

    def __post_init__(self):
        super().__post_init__()
        if self.optimizer_scheduler_args is None:
            # One optimizer (AdamW) with two parameter groups, each with its own
            # kwargs (order-matched to OptimizerConfig.params).
            self.optimizer_scheduler_args = [
                {
                    "optimizer_cls": "AdamW",
                    "optimizer_kwargs": [
                        {
                            "lr": self.q_proj_lr},
                        {
                            "lr": self.non_q_proj_lr},],},]


class TwoGroupExampleTrainer(GeneralizedTrainer):
    training_args_cls = TwoGroupExampleTrainingArguments
    optimizer_setup = staticmethod(_build_two_group_optimizer_setup)


TRAINER_REGISTRY.register("two_group_optimizer_trainer")(TwoGroupExampleTrainer)
