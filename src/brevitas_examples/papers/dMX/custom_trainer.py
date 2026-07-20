# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
from typing import List
from typing import Optional

import numpy as np
import torch
from transformers import TrainerCallback

from brevitas.loss.weighted_bit_width import ActivationFloatBitWidthWeightedBySize
from brevitas.loss.weighted_bit_width import BitWidthWeighted
from brevitas.loss.weighted_bit_width import WeightFloatBitWidthWeightedBySize
import brevitas.nn as qnn
from brevitas.quant_tensor import QuantTensor
from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrices
from brevitas.utils.quant_utils import has_learned_activation_bit_width
from brevitas.utils.quant_utils import has_learned_weight_bit_width
from brevitas_examples.llm.llm_quant.rotation_optimization import GeneralizedTrainer
from brevitas_examples.llm.llm_quant.trainer_utils import TRAINER_REGISTRY
from brevitas_examples.llm.llm_quant.trainer_utils import TrainingArguments


# ---------------------------------------------------------------------------
# Bit-width average criteria (unweighted by tensor size)
# ---------------------------------------------------------------------------
class WeightFloatBitWidthAverage(BitWidthWeighted):

    def __init__(self, model):
        super(WeightFloatBitWidthAverage, self).__init__(model=model)

    def register_hooks(self):

        def hook_fn(module, input, output: QuantTensor):
            self.weighted_bit_width_list.append(
                (output.mantissa_bit_width + output.exponent_bit_width + int(output.signed)).to(
                    torch.float32))
            self.tot_num_elements += 1

        for name, module in self.model.named_modules():
            if has_learned_weight_bit_width(module):
                h = module.register_forward_hook(hook_fn)
                self.list_of_hooks.append(h)


class ActivationFloatBitWidthAverage(BitWidthWeighted):

    def __init__(self, model):
        super(ActivationFloatBitWidthAverage, self).__init__(model=model)

    def register_hooks(self):

        def hook_fn(module, input, output: QuantTensor):
            self.weighted_bit_width_list.append(
                (output.mantissa_bit_width + output.exponent_bit_width + int(output.signed)).to(
                    torch.float32))
            self.tot_num_elements += 1

        for name, module in self.model.named_modules():
            if has_learned_activation_bit_width(module):
                h = module.register_forward_hook(hook_fn)
                self.list_of_hooks.append(h)


# ---------------------------------------------------------------------------
# Temperature annealing callback
# ---------------------------------------------------------------------------
class TemperatureAnnealingCallback(TrainerCallback):
    """Anneal a ``temperature`` attribute on model modules.

    Uses an exponential decay schedule (reversed, so temperature increases
    from ``start`` to ``end`` over training) to sharpen discrete
    selections in Gumbel-softmax / straight-through estimator modules.

    Parameters
    ----------
    delay_start : float
        Fraction of ``max_steps`` to wait before starting the anneal.
    max_steps : int
        Total number of training steps.
    start : float
        Initial temperature value (high = soft).
    end : float
        Final temperature value (low = sharp).
    """

    def __init__(
            self, delay_start: float, max_steps: int, start: float = 20., end: float = 0.4) -> None:
        super().__init__()
        self.start_step = int(delay_start * max_steps)
        anneal_steps = max_steps - self.start_step

        def _exp_decay(x, s=start, e=end, duration=1.):
            k = -np.log(e / s) / duration
            return s * np.exp(-k * x)

        x = np.linspace(0, 1, max(anneal_steps, 1))
        # Reverse so temperature goes from *end* (0.4) up to *start* (20)
        self.temperature_values = _exp_decay(x)[::-1].copy()

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        step = state.global_step
        idx = step - self.start_step
        if 0 <= idx < len(self.temperature_values):
            with torch.no_grad():
                for m in model.modules():
                    if hasattr(m, "temperature"):
                        m.temperature.value.fill_(self.temperature_values[idx])


# ---------------------------------------------------------------------------
# Bit-width tying helper
# ---------------------------------------------------------------------------
def _tie_bit_widths(model: torch.nn.Module, args) -> None:
    """Tie mantissa and exponent bit-width offsets within each ``QuantLinear``.

    For every ``QuantLinear`` module with learned bit-width parameters:

    1. The **activation** exponent ``bit_width_offset`` is set to point
       to the same tensor as the activation mantissa ``bit_width_offset``.
    2. The **weight** mantissa ``bit_width_offset`` is tied to the
       activation mantissa ``bit_width_offset``.
    3. The **weight** exponent ``bit_width_offset`` is tied to the
       weight mantissa ``bit_width_offset``.

    This ensures a single learnable offset controls all four
    (act mantissa, act exponent, weight mantissa, weight exponent)
    bit-width offsets per layer.
    """
    for m in model.modules():
        if not isinstance(m, qnn.QuantLinear):
            continue
        act_tq = m.input_quant.fused_activation_quant_proxy.tensor_quant
        weight_tq = m.weight_quant.tensor_quant
        if not hasattr(act_tq.mantissa_bit_width_impl, "bit_width_offset"):
            continue
        # Get the activation mantissa offset -- this is the master parameter.
        act_mantissa_offset = act_tq.mantissa_bit_width_impl.bit_width_offset
        # Tie activation exponent to activation mantissa
        if hasattr(act_tq.exponent_bit_width_impl, "bit_width_offset"):
            act_tq.exponent_bit_width_impl.bit_width_offset = act_mantissa_offset
        # Tie weight mantissa to activation mantissa
        weight_tq.mantissa_bit_width_impl.bit_width_offset = act_mantissa_offset
        # Tie weight exponent to weight mantissa
        if hasattr(weight_tq.exponent_bit_width_impl, "bit_width_offset"):
            weight_tq.exponent_bit_width_impl.bit_width_offset = act_mantissa_offset


# ---------------------------------------------------------------------------
# Parameter selectors for optimizer configs
# ---------------------------------------------------------------------------
def _get_rotation_params(
        model: torch.nn.Module,
        training_args: "RotationLearnedBitWidthTrainingArguments") -> List[torch.nn.Parameter]:
    """Return trainable rotation matrices for CaileySGD optimisation."""
    return extract_trainable_rotation_matrices(model)


def _get_bit_width_params(
        model: torch.nn.Module,
        training_args: "RotationLearnedBitWidthTrainingArguments") -> List[torch.nn.Parameter]:
    """Tie bit-widths and return the (now unique) bit-width parameters.

    Bit-width tying is performed here because this callable runs during
    optimizer construction, before the trainer is instantiated -- the
    earliest point at which the model is available and parameters can
    be safely shared.
    """
    _tie_bit_widths(model, training_args)
    # After tying, collect unique bit-width parameters
    seen: set = set()
    params: List[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if "bit_width" in n and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return params


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------
@dataclass
class RotationLearnedBitWidthTrainingArguments(TrainingArguments):
    """Training arguments for rotation + learned float bit-width optimisation.

    Inherits all fields from ``TrainingArguments`` (which itself extends
    ``transformers.TrainingArguments``).  Adds bit-width-specific knobs
    and sensible defaults for joint rotation + bit-width optimisation.

    Unlike the full learned-float-bitwidth trainer, this configuration:
    * Does **not** fine-tune model weights (no AdamW group).
    * Uses the **default CE loss** (no distillation).
    """

    # Override defaults from parent
    optimizer_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Data type for CaileySGD optimizer computations (e.g. 'float32'). "
                "None means use the parameter dtype."})

    # Bit-width regularisation
    target_bit_width: float = field(
        default=0.,
        metadata={
            "help":
                "Target average bit-width for the regularisation penalty. "
                "0 disables the penalty."})
    delay_start: float = field(
        default=0.6,
        metadata={
            "help":
                "Fraction of max_steps to wait before starting temperature annealing "
                "(value in [0, 1])."})

    # CaileySGD hyper-parameters (for rotation matrices)
    rotation_lr: float = field(
        default=1.5, metadata={"help": "Learning rate for CaileySGD (rotation matrices)."})

    # SGD hyper-parameters (for bit-width parameters)
    bw_learning_rate: float = field(
        default=1.,
        metadata={"help": "Learning rate for the SGD optimiser on bit-width parameters."})

    simple_average_loss: bool = field(
        default=True,
        metadata={
            "help":
                "If True, use simple average bit-width for loss. Otherwise, use weighted average"})

    def __post_init__(self):
        super().__post_init__()
        self.target_bit_width = torch.tensor(self.target_bit_width)
        if self.optimizer_scheduler_args is None:
            self.optimizer_scheduler_args = [
                # Optimizer 0: CaileySGD for rotation matrices
                {
                    "optimizer_cls": "CaileySGD",
                    "param_setup": [{
                        "get_param_fn": _get_rotation_params,
                        "optimizer_kwargs": {
                            "lr": self.rotation_lr,
                            "stiefel": True,
                            "dtype": self.optimizer_dtype,},}],
                    "scheduler_cls": "ConstantLR",
                    "scheduler_kwargs": {
                        "factor": 1.}},
                # Optimizer 1: SGD for bit-width parameters
                {
                    "optimizer_cls": "SGD",
                    "param_setup": [{
                        "get_param_fn": _get_bit_width_params,
                        "optimizer_kwargs": {
                            "lr": self.bw_learning_rate,
                            "momentum": 0.99,
                            "weight_decay": 0.,
                            "nesterov": True,},}],
                    "scheduler_cls": "ConstantLR",
                    "scheduler_kwargs": {
                        "factor": 1.}},]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class RotationLearnedBitWidthTrainer(GeneralizedTrainer):
    """Trainer for joint rotation and learned float bit-width optimisation.

    Extends ``GeneralizedTrainer`` with:

    * **Bit-width regularisation loss** penalising the average learned
      float bit-width when it exceeds the target ``target_bit_width``.
    * **Temperature annealing** via ``TemperatureAnnealingCallback``.
    * Automatic hook cleanup after training.

    Model weights are frozen; only rotation matrices and bit-width
    parameters are optimised.  Default CE loss is used (no distillation).
    """

    training_args_cls = RotationLearnedBitWidthTrainingArguments

    def __init__(self, args: RotationLearnedBitWidthTrainingArguments = None, **kwargs) -> None:
        super().__init__(args=args, **kwargs)
        self.target_bit_width = args.target_bit_width

        # Bit-width loss criteria -- register forward hooks to accumulate
        # averaged bit-width values during each forward pass.
        if args.simple_average_loss:
            self.weight_criterion = WeightFloatBitWidthAverage(self.model)
            self.act_criterion = ActivationFloatBitWidthAverage(self.model)
        else:
            self.weight_criterion = WeightFloatBitWidthWeightedBySize(self.model)
            self.act_criterion = ActivationFloatBitWidthWeightedBySize(self.model)

        # Initialise temperature on all modules that support it
        with torch.no_grad():
            for m in self.model.modules():
                if hasattr(m, "temperature"):
                    m.temperature.value.fill_(0.4)

        # Add temperature annealing callback
        if args.max_steps > 0:
            self.add_callback(
                TemperatureAnnealingCallback(
                    delay_start=args.delay_start, max_steps=args.max_steps))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute CE loss with bit-width regularisation.

        Delegates to ``GeneralizedTrainer.compute_loss`` for the base CE
        loss, then adds the bit-width penalty.
        """
        # Reset accumulators before the forward pass
        self.weight_criterion.zero_accumulated_values()
        self.act_criterion.zero_accumulated_values()

        # Parent handles CE loss (distillation is disabled by default)
        result = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        loss, outputs = result

        # Bit-width regularisation
        try:
            weight_bw = self.weight_criterion.retrieve()
            act_bw = self.act_criterion.retrieve()
        except Exception:
            # No elements accumulated (e.g. no learned bit-width modules)
            weight_bw = torch.tensor(0.)
            act_bw = torch.tensor(0.)

        # Log average bit-widths via HF Trainer's built-in logging

        # Penalty is non-zero only when the average bit-width exceeds
        # the target (target_bit_width).  The factor of 5 controls the
        # penalty slope.
        zero = torch.tensor(0., device=loss.device)
        current_avg_bw = (weight_bw + act_bw) / 2.
        target_bw = self.target_bit_width.to(loss.device)
        bw_penalty = torch.max(zero, 5. * (current_avg_bw - target_bw))

        self.log({
            "weight_bw": weight_bw.detach().item(),
            "act_bw": act_bw.detach().item(),
            "penalty": bw_penalty.detach().item()})
        loss = loss + bw_penalty

        return (loss, outputs) if return_outputs else loss

    def train(self, *args, **kwargs):
        """Train and clean up bit-width criterion hooks afterwards."""
        result = super().train(*args, **kwargs)

        # Print learned bit-widths per quantised layer
        self.model.eval()
        for n, m in self.model.named_modules():
            if isinstance(m, qnn.QuantLinear):
                act_tq = m.input_quant.fused_activation_quant_proxy.tensor_quant
                weight_tq = m.weight_quant.tensor_quant
                print(
                    n,
                    act_tq.exponent_bit_width_impl(),
                    act_tq.mantissa_bit_width_impl(),
                    weight_tq.exponent_bit_width_impl(),
                    weight_tq.mantissa_bit_width_impl())
            elif isinstance(m, qnn.QuantScaledDotProductAttention):
                sdpa_tq = (m.k_transposed_quant.act_quant.fused_activation_quant_proxy.tensor_quant)
                print(n, sdpa_tq.exponent_bit_width_impl(), sdpa_tq.mantissa_bit_width_impl())

        self.weight_criterion.remove_hooks()
        self.act_criterion.remove_hooks()
        return result


# ---------------------------------------------------------------------------
# Register the trainer under the name "rotation_learned_bitwidth".
#
# The trainer exposes its training-arguments class (which in turn defines the
# optimizer setup via ``optimizer_scheduler_args``) through the
# ``training_args_cls`` class attribute, so a single registry entry is enough.
# ---------------------------------------------------------------------------
TRAINER_REGISTRY.register("rotation_learned_bitwidth")(RotationLearnedBitWidthTrainer)
