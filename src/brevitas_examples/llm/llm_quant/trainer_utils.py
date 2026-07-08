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
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
import transformers
from transformers import Trainer

from brevitas.utils.python_utils import Registry
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
    model: torch.nn.Module,
    training_args: transformers.TrainingArguments,
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
