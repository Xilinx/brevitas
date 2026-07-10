# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
import transformers
from transformers import get_scheduler
from transformers import Trainer

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.trainer_utils import parse_lr_scheduler_class
from brevitas_examples.common.trainer_utils import parse_optimizer_class

# A parameter-selection callable for a single parameter group:
# ``(model, training_args) -> List[Parameter]``. Each ``optimizer_scheduler_args``
# entry carries a ``param_setup`` list whose per-group dicts each hold one of
# these under ``get_param_fn``.
ParamsFn = Callable[[torch.nn.Module, "transformers.TrainingArguments"], List[torch.nn.Parameter]]

# Single registry for out-of-source customization of the training process.
# Users register a custom Trainer class under a config name via a plugin .py
# file. The Trainer class may expose a ``training_args_cls`` class attribute to
# customise the training arguments (including the optimizer/scheduler setup via
# ``optimizer_scheduler_args``); when left at its default the built-in behaviour
# of the LLM example is used.
TRAINER_REGISTRY = Registry[Type[Trainer]](registry_name="TrainerRegistry")


class MultiOptimizer(torch.optim.Optimizer):
    """Wrapper to handle multiple optimizers as a single optimizer for Trainer.

    Allows attaching different optimizer/scheduler pairs to different parameter
    groups (e.g. CaileySGD for rotation matrices and AdamW for other params).

    Inherits from :class:`torch.optim.Optimizer` (without calling
    ``super().__init__()``) so that ``isinstance`` checks in ``accelerate``
    and the HuggingFace ``Trainer`` recognise this object as an optimizer.

    .. note::
        The HuggingFace ``Trainer`` calls ``model.zero_grad()`` rather than
        ``optimizer.zero_grad()``, so :meth:`zero_grad` is typically **not**
        invoked during training.  Sub-optimizers that perform bookkeeping
        inside ``zero_grad()`` beyond clearing ``.grad`` should be aware of
        this.
    """

    def __init__(self, optimizers: List[torch.optim.Optimizer]) -> None:
        # Intentionally skip super().__init__() — this is a thin wrapper
        # that delegates all real work to the sub-optimizers.
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = False) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Optional[Callable[[], Any]] = None) -> Optional[Any]:
        # If a closure is provided, execute it exactly once before stepping
        # any sub-optimizer.  Passing the closure to every sub-optimizer would
        # execute it N times (one full forward+backward per optimizer), which
        # doubles compute and corrupts accumulated gradients.
        loss = None
        if closure is not None:
            loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    @property
    def state(self) -> Dict[torch.nn.Parameter, Any]:
        # Returns a **snapshot** (shallow copy) of the merged optimizer
        # states.  Mutations to this dict do not propagate back to the
        # sub-optimizers.  Keys are parameter objects; if two sub-optimizers
        # manage the same parameter (a misconfiguration), the later entry
        # silently wins — detect and raise to prevent silent corruption.
        merged: Dict[str, Any] = {}
        for optimizer in self.optimizers:
            for k, v in optimizer.state.items():
                if k in merged:
                    raise RuntimeError(
                        f"MultiOptimizer.state: parameter {k} appears in "
                        "multiple sub-optimizers.  Each parameter must belong "
                        "to exactly one optimizer.")
                merged[k] = v
        return merged

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        return [
            param_group for optimizer in self.optimizers for param_group in optimizer.param_groups]

    @property
    def defaults(self) -> Dict[str, Any]:
        # Return the defaults of the first sub-optimizer as a best-effort
        # approximation.  This is accessed by accelerate's
        # AcceleratedOptimizer property delegation.
        if self.optimizers:
            return self.optimizers[0].defaults
        return {}

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisation-safe state dict for all sub-optimizers."""
        return {"sub_optimizer_states": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore state from a dict produced by :meth:`state_dict`."""
        sub_states = state_dict.get("sub_optimizer_states")
        if sub_states is None:
            raise ValueError(
                "MultiOptimizer.load_state_dict expects a dict with key "
                "'sub_optimizer_states' containing a list of per-optimizer "
                "state dicts.")
        if len(sub_states) != len(self.optimizers):
            raise ValueError(
                f"MultiOptimizer.load_state_dict: expected "
                f"{len(self.optimizers)} sub-optimizer state dicts, "
                f"got {len(sub_states)}.")
        for optimizer, sub_state in zip(self.optimizers, sub_states):
            optimizer.load_state_dict(sub_state)


class MultiScheduler:
    """Wrapper to handle multiple schedulers as a single scheduler for Trainer.

    Schedulers in the list may be ``None`` to indicate no scheduling for the
    corresponding optimizer.

    Serialisation format
    --------------------
    :meth:`state_dict` returns::

        {"sub_scheduler_states": [state_dict_or_none, ...]}

    :meth:`load_state_dict` expects the same structure.
    """

    def __init__(self, schedulers: List[Optional[LRScheduler]]) -> None:
        self.schedulers = schedulers if schedulers else []

    def step(self, *args: Any, **kwargs: Any) -> None:
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step(*args, **kwargs)

    def get_last_lr(self) -> List[float]:
        """Return the concatenation of all schedulers' ``get_last_lr()`` lists.

        ``None`` entries are skipped so that the first real LR occupies
        index 0 — which is the index the HuggingFace Trainer reads for
        logging.
        """
        lrs: List[float] = []
        for scheduler in self.schedulers:
            if scheduler is not None:
                lrs.extend(scheduler.get_last_lr())
        return lrs

    def state_dict(self) -> Dict[str, Any]:
        """Return a serialisation-safe state dict for all sub-schedulers."""
        return {
            "sub_scheduler_states": [
                scheduler.state_dict() if scheduler is not None else None
                for scheduler in self.schedulers]}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore state from a dict produced by :meth:`state_dict`.

        Validates the format and length before applying.
        """
        if not isinstance(state_dict, dict) or "sub_scheduler_states" not in state_dict:
            raise ValueError(
                "MultiScheduler.load_state_dict expects a dict with key "
                "'sub_scheduler_states' containing a list of per-scheduler "
                "state dicts (or None entries).")
        sub_states = state_dict["sub_scheduler_states"]
        if len(sub_states) != len(self.schedulers):
            raise ValueError(
                f"MultiScheduler.load_state_dict: expected "
                f"{len(self.schedulers)} sub-scheduler state dicts, "
                f"got {len(sub_states)}.")
        for scheduler, sub_state in zip(self.schedulers, sub_states):
            if scheduler is not None and sub_state is not None:
                scheduler.load_state_dict(sub_state)


def _resolve_params(
        params_fn: ParamsFn, model: torch.nn.Module,
        training_args: transformers.TrainingArguments) -> List[torch.nn.Parameter]:
    """Resolve a single parameter-selection callable into a list of parameters.

    *params_fn* is a callable ``(model, training_args) -> List[Parameter]``. The
    selected parameters have ``requires_grad`` enabled.
    """
    params = list(params_fn(model, training_args))
    for param in params:
        param.requires_grad = True
    return params


def _build_optimizers_from_configs(
    model: torch.nn.Module, training_args: transformers.TrainingArguments
) -> Tuple[Union[Optimizer, MultiOptimizer], Optional[Union[LRScheduler, MultiScheduler]]]:
    """Build a ``(MultiOptimizer, MultiScheduler | None)`` pair from
    ``training_args.optimizer_scheduler_args``: a list with one entry per
    optimizer, each fully self-contained. Each entry may contain:

    * ``optimizer_cls`` – optimizer class *name* (str), resolved against the
      optimizer namespaces (default: ``"CaileySGD"``).
    * ``param_setup`` – a list of per-parameter-group dicts, each with a
      ``get_param_fn`` (a callable ``(model, training_args) -> List[Parameter]``)
      and an ``optimizer_kwargs`` dict of that group's kwargs.
    * ``scheduler_cls`` – optional LR scheduler class *name* (str).
    * ``scheduler_kwargs`` – optional dict of kwargs for the scheduler.
    """
    optimizers: List[Optimizer] = []
    schedulers: List[Optional[LRScheduler]] = []

    os_args: List[Dict[str, Any]] = training_args.optimizer_scheduler_args

    for entry in os_args:
        # Build the parameter groups for this optimizer (one per param_setup
        # entry), attaching each group's kwargs.
        param_groups = [{
            "params": _resolve_params(group["get_param_fn"], model, training_args),
            **group["optimizer_kwargs"]} for group in entry["param_setup"]]

        # Resolve the optimizer class from its string name.
        optimizer_cls_name = entry.get("optimizer_cls")
        optimizer_class = (
            parse_optimizer_class(optimizer_cls_name)
            if isinstance(optimizer_cls_name, str) else optimizer_cls_name)
        optimizer = optimizer_class(param_groups)
        optimizers.append(optimizer)

        # Resolve the optional scheduler class from its string name.
        scheduler_cls_name = entry.get("scheduler_cls", None)
        if scheduler_cls_name is not None:
            scheduler_class = (
                parse_lr_scheduler_class(scheduler_cls_name)
                if isinstance(scheduler_cls_name, str) else scheduler_cls_name)
            scheduler_kwargs = entry.get("scheduler_kwargs", {})
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
            schedulers.append(scheduler)
        else:
            scheduler = None
            schedulers.append(scheduler)

    if len(optimizers) > 1:
        multi_optimizer = MultiOptimizer(optimizers)
        # Always return a MultiScheduler, even when all entries are None.
        # This prevents the HF Trainer from creating its own scheduler
        # (which would fail because MultiOptimizer is not a real Optimizer).
        # Entries left as None are filled in later by the Trainer's
        # ``create_scheduler`` override with the HuggingFace default scheduler
        # (which requires ``num_training_steps``, unavailable here). MultiScheduler
        # tolerates None entries until then.
        multi_scheduler = MultiScheduler(schedulers)
        return multi_optimizer, multi_scheduler
    else:
        # A None scheduler here lets the HF Trainer build its default scheduler.
        return optimizers[0], schedulers[0]


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # By default, arguments are saved in the current working directory
    output_dir: Optional[str] = field(default=os.getcwd())
    # NOTE: Currently, there is no infrastructure to resume training
    # from a checkpoint, so related files are not save by default
    save_strategy: Optional[str] = field(default="no")

    ### Optimizer args
    optimizer_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Data type for CaileySGD optimizer computations. None means use parameter dtype."})

    ### Multi-optimizer/scheduler args
    # List of dicts, one self-contained entry per optimizer.  Each dict may
    # contain:
    #   * 'optimizer_cls'    : optimizer class *name* (str), resolved against
    #                          the optimizer namespaces. Defaults to CaileySGD.
    #   * 'param_setup'      : a list of per-parameter-group dicts, each with a
    #                          'get_param_fn' (callable
    #                          ``(model, training_args) -> List[Parameter]``) and
    #                          an 'optimizer_kwargs' dict for that group.
    #   * 'scheduler_cls'    : optional LR scheduler class *name* (str).
    #   * 'scheduler_kwargs' : optional dict of kwargs for the scheduler.
    optimizer_scheduler_args: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help":
                "List of dicts, one per optimizer. Each dict may contain "
                "'optimizer_cls' (str), 'param_setup' (list of per-group dicts, "
                "each with 'get_param_fn' callable and 'optimizer_kwargs' dict), "
                "'scheduler_cls' (str) and 'scheduler_kwargs' (dict)."})

    ### Distillation Loss args
    use_distillation_loss: bool = field(
        default=False, metadata={"help": "Whether to compute the distillation loss."})
    gamma: float = field(
        default=1., metadata={"help": "Gamma balances CE loss (gamma) vs KD loss (1-gamma)."})
    temperature: float = field(
        default=1.0, metadata={"help": "Softmax temperature for the soft targets"})
    # Considering the huge vocabulary size of LLMs, it could be better selecting only the first K
    # labels when using the distillation loss
    topk: int = field(
        default=-1,
        metadata={"help": "Consider the first K logits when computing distillation loss"})
    kl_loss_reduction: str = field(
        default="batchmean", metadata={"help": "Reduction mode to use when computing KL loss"})


class GeneralizedTrainer(Trainer):

    # Training-arguments class consumed by the LLM entrypoint when this trainer
    # is registered via ``--custom-trainer``. Subclasses may override it to
    # customise the training arguments (including the optimizer/scheduler setup
    # exposed through ``optimizer_scheduler_args``). When left at the built-in
    # ``TrainingArguments``, the default behaviour of the LLM example is used.
    training_args_cls: Type[transformers.TrainingArguments] = TrainingArguments

    def __init__(
            self,
            args: Optional[TrainingArguments] = None,
            teacher_model: Optional[torch.nn.Module] = None,
            **kwargs: Any) -> None:
        super().__init__(args=args, **kwargs)
        self.use_distillation_loss = args.use_distillation_loss
        self.gamma = args.gamma
        self.temperature = args.temperature
        self.kl_loss_reduction = args.kl_loss_reduction
        self.teacher_model = None if teacher_model is None else offload_model(teacher_model)

    def _default_scheduler(self, optimizer, num_training_steps):
        """Build the HuggingFace default LR scheduler for a single optimizer.

        Mirrors :meth:`transformers.Trainer.create_scheduler`: honours
        ``lr_scheduler_type`` (default ``"linear"``), the (possibly ratio-based)
        warmup steps and any ``lr_scheduler_kwargs``.
        """
        return get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
        )

    def create_scheduler(self, num_training_steps, optimizer=None):
        """Set up the LR scheduler, matching the HuggingFace default.

        The optimizer/scheduler pair is built eagerly (before the Trainer exists)
        by :func:`_build_optimizers_from_configs`, which cannot know
        ``num_training_steps``. Any optimizer left without an explicit scheduler
        is therefore represented by a ``None`` placeholder and filled in here with
        the HuggingFace default scheduler (linear warmup + linear decay).

        Handles both the single-optimizer case (``self.lr_scheduler is None``,
        delegated to the base implementation) and the multi-optimizer case
        (a :class:`MultiScheduler` with ``None`` entries).
        """
        # Multi-optimizer case: fill in any None sub-schedulers in place.
        if isinstance(self.lr_scheduler, MultiScheduler) and isinstance(self.optimizer,
                                                                        MultiOptimizer):
            for idx, (sub_optimizer, sub_scheduler) in enumerate(zip(self.optimizer.optimizers,
                                                                     self.lr_scheduler.schedulers)):
                if sub_scheduler is None:
                    self.lr_scheduler.schedulers[idx] = self._default_scheduler(
                        sub_optimizer, num_training_steps)
            return self.lr_scheduler
        # Single-optimizer case: the base implementation already builds the
        # HuggingFace default scheduler when self.lr_scheduler is None.
        return super().create_scheduler(num_training_steps, optimizer)

    @staticmethod
    def forward_kl_loss(
            student_logits, teacher_logits, temperature=1.0, topk=-1, reduction="batchmean"):
        out_dtype = student_logits.dtype
        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if topk > 0:
            teacher_log_probs, indices = teacher_log_probs.topk(topk, dim=-1, sorted=False)
            student_log_probs = student_log_probs.gather(-1, indices)
            # After selecting the top-k entries, the log-probabilities no longer
            # sum to one over the truncated vocabulary. Renormalize them via
            # logsumexp so they form valid log-probability distributions over
            # the selected subset, consistent with the log_target=True KL below.
            student_log_probs = student_log_probs - torch.logsumexp(
                student_log_probs, dim=-1, keepdim=True)
            teacher_log_probs = teacher_log_probs - torch.logsumexp(
                teacher_log_probs, dim=-1, keepdim=True)

        loss = F.kl_div(student_log_probs, teacher_log_probs, reduction=reduction, log_target=True)
        if reduction == "none":
            # We sum across the vocabulary dim, and then average the rest
            loss = loss.sum(dim=-1).mean()
        return loss.to(out_dtype)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # If distillation loss is used, we need to retrieve the original model's outputs
        distillation_return_outputs = return_outputs if not self.use_distillation_loss else True

        loss = super().compute_loss(model, inputs, distillation_return_outputs, num_items_in_batch)

        if distillation_return_outputs:
            loss, outputs = loss

        if self.use_distillation_loss:
            with torch.no_grad(), quantization_status_manager(self.teacher_model, disable_act_quant=True, disable_weight_quant=True, disable_bias_quant=True):
                fp_outputs = self.teacher_model(**inputs)
            # Compute the distillation loss
            distill_loss = GeneralizedTrainer.forward_kl_loss(
                student_logits=outputs.logits,
                teacher_logits=fp_outputs.logits,
                temperature=self.temperature,
                reduction=self.kl_loss_reduction,
                topk=self.args.topk)

            if (self.args.average_tokens_across_devices and
                (self.model_accepts_loss_kwargs or self.compute_loss_func) and
                    num_items_in_batch is not None):
                distill_loss = distill_loss * self.accelerator.num_processes
            loss = self.gamma * loss + (1. - self.gamma) * distill_loss

        return (loss, outputs) if return_outputs else loss
