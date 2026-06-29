# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import torch
from torch.optim.lr_scheduler import ConstantLR
import transformers
from transformers import Trainer

from brevitas.utils.python_utils import Registry
from brevitas_examples.common.learned_round.learned_round_args import parse_lr_scheduler_class
from brevitas_examples.common.learned_round.learned_round_args import parse_optimizer_class

# Default optimizer class name, resolved against ``OPTIMIZER_NAMESPACES`` from
# ``learned_round_args`` (which include ``torch.optim`` and ``brevitas.optim``).
DEFAULT_OPTIMIZER_CLS = "CaileySGD"

ParamsFn = Callable[[torch.nn.Module, "transformers.TrainingArguments"], List[torch.nn.Parameter]]


@dataclass
class OptimizerConfig:
    """Configuration describing which parameters an optimizer should train.

    Only the parameter-selection part of an optimizer setup lives here, since
    it cannot be expressed as a CLI/string value. Everything else (the
    optimizer/scheduler class names and their kwargs) is supplied through
    ``TrainingArguments.optimizer_scheduler_args``, order-matched to the list
    of :class:`OptimizerConfig` instances.

    Parameters
    ----------
    params : callable or list of callables
        Each callable ``(model, training_args) -> List[Parameter]`` selects the
        parameters of one parameter group handled by the optimizer. The
        parameters are selected lazily, after the model has been prepared for
        fine-tuning. A single optimizer may handle multiple parameter groups,
        each with its own per-group ``optimizer_kwargs`` entry in
        ``TrainingArguments.optimizer_scheduler_args``.

        For convenience a single callable may be passed directly; it is
        normalised to a one-element list. After ``__post_init__`` ``params`` is
        always a list of callables with at least one element.
    """
    params: Union[ParamsFn, List[ParamsFn]]

    def __post_init__(self) -> None:
        self.params = self._normalize_params(self.params)

    @staticmethod
    def _normalize_params(params: Any) -> List[ParamsFn]:
        # A single callable selects one group -> wrap into a one-element list.
        if callable(params):
            return [params]
        # Multiple groups must be provided as a list (never a tuple).
        if not isinstance(params, list):
            raise TypeError(
                "OptimizerConfig.params must be a callable or a list of "
                "callables (one per parameter group).")
        if len(params) == 0:
            raise ValueError("OptimizerConfig.params must not be empty.")
        if not all(callable(element) for element in params):
            raise TypeError(
                "OptimizerConfig.params must be a callable or a list of "
                "callables (one per parameter group).")
        return params


# User-facing parameter-selection spec for a single optimizer: either one
# selection callable (single parameter group) or a list of callables (one per
# parameter group). Users return a list of these (one per optimizer) from a
# trainer's ``optimizer_setup``; they are wrapped into :class:`OptimizerConfig`
# internally (see :func:`_to_optimizer_configs`).
OptimizerParamsSpec = Union[ParamsFn, List[ParamsFn]]


def _to_optimizer_configs(optimizer_setup: List[OptimizerParamsSpec]) -> List[OptimizerConfig]:
    """Wrap a user-provided optimizer setup into a list of ``OptimizerConfig``.

    ``optimizer_setup`` is a list with one entry per optimizer; each entry is a
    parameter-selection callable or a list of such callables (one per parameter
    group). An entry that is already an :class:`OptimizerConfig` is passed
    through unchanged.
    """
    if not isinstance(optimizer_setup, list):
        raise TypeError(
            "optimizer_setup must be a list with one entry per optimizer, got "
            f"{type(optimizer_setup)}.")
    return [
        entry if isinstance(entry, OptimizerConfig) else OptimizerConfig(params=entry)
        for entry in optimizer_setup]


# Single registry for out-of-source customization of the training process.
# Users register a custom Trainer class under a config name via a plugin .py
# file. The Trainer class may expose ``training_args_cls`` and
# ``optimizer_setup`` class attributes to customise the training arguments and
# optimizer/scheduler setup; when these are left at their defaults the built-in
# behaviour of the LLM example is used.
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

    def step(self, closure=None):
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
    def state(self) -> Dict[str, Any]:
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

    def __init__(self, schedulers: List[Optional[Any]]) -> None:
        self.schedulers = schedulers if schedulers else []

    def step(self, *args, **kwargs) -> None:
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
    selected parameters have ``requires_grad`` enabled.  Raises ``RuntimeError``
    if no parameters are selected, since an empty parameter group is always a
    misconfiguration.
    """
    params = list(params_fn(model, training_args))
    if len(params) == 0:
        raise RuntimeError(
            "A parameter-selection function returned no parameters. Each "
            "parameter group must contain at least one parameter.")
    for param in params:
        param.requires_grad = True
    return params


def _build_optimizer_param_groups(
        config: "OptimizerConfig",
        optimizer_kwargs: List[Dict[str, Any]],
        model: torch.nn.Module,
        training_args: transformers.TrainingArguments) -> List[Dict[str, Any]]:
    """Build the list of PyTorch parameter groups for a single optimizer.

    ``config.params`` is always a list of parameter selectors (one per group;
    see :class:`OptimizerConfig`). *optimizer_kwargs* is the order-matched list
    of per-group kwargs dicts of the same length. Every parameter group requires
    its own kwargs: each entry must be a non-empty dict.
    """
    params = config.params
    # 'optimizer_kwargs' is always a list with one non-empty dict per parameter
    # group. A single dict (or None) is a misconfiguration.
    if not isinstance(optimizer_kwargs, list):
        raise RuntimeError(
            "'optimizer_kwargs' must be a list with one kwargs dict per "
            f"parameter group, got {type(optimizer_kwargs)}.")
    if len(optimizer_kwargs) != len(params):
        raise RuntimeError(
            f"Number of parameter groups ({len(params)}) does not match the "
            f"number of per-group 'optimizer_kwargs' ({len(optimizer_kwargs)}).")

    param_groups = []
    for params_fn, group_kwargs in zip(params, optimizer_kwargs):
        if not isinstance(group_kwargs, dict) or len(group_kwargs) == 0:
            raise RuntimeError(
                "Each entry in 'optimizer_kwargs' must be a non-empty dict; "
                f"got {group_kwargs!r}.")
        group_params = _resolve_params(params_fn, model, training_args)
        param_groups.append({"params": group_params, **group_kwargs})
    return param_groups


def _build_optimizers_from_configs(
        model: torch.nn.Module,
        training_args: transformers.TrainingArguments,
        optimizer_configs: List["OptimizerConfig"]) -> tuple:
    """Build a ``(MultiOptimizer, MultiScheduler | None)`` pair from a
    list of :class:`OptimizerConfig` instances.

    Each :class:`OptimizerConfig` only carries ``params`` (the parameter
    selection). The optimizer/scheduler class names and their kwargs are read
    from ``training_args.optimizer_scheduler_args[i]``, an order-matched list of
    dicts. Each dict may contain:

    * ``optimizer_cls`` – optimizer class *name* (str), resolved against the
      optimizer namespaces (default: ``"CaileySGD"``).
    * ``optimizer_kwargs`` – a list with one non-empty kwargs dict per parameter
      group, order-matched to ``OptimizerConfig.params``. Every group requires
      its own kwargs: a bare dict, ``None``, or empty/None entries are rejected.
    * ``scheduler_cls`` – optional LR scheduler class *name* (str).
    * ``scheduler_kwargs`` – optional dict of kwargs for the scheduler.

    If ``training_args.optimizer_scheduler_args`` is ``None`` or its length does
    not match *optimizer_configs*, this fails.
    """
    optimizers: List[torch.optim.Optimizer] = []
    schedulers: List[Optional[Any]] = []

    os_args: Optional[List[Dict[str,
                                Any]]] = getattr(training_args, "optimizer_scheduler_args", None)
    if os_args is None or len(os_args) != len(optimizer_configs):
        raise RuntimeError("Scheduler/Optimizer arguments do not match the configs")

    for i, config in enumerate(optimizer_configs):
        # Look up the args from the order-matched training args list
        entry = os_args[i]

        # Build the parameter groups for this optimizer (one or many).
        param_groups = _build_optimizer_param_groups(
            config, entry.get("optimizer_kwargs"), model, training_args)

        # Resolve the optimizer class from its string name.
        optimizer_cls_name = entry.get("optimizer_cls", DEFAULT_OPTIMIZER_CLS)
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
        # If no scheduler is specified, we use a "dummy" scheduler with constant lr
        schedulers = [
            ConstantLR(optimizer, factor=1.) if scheduler is None else scheduler
            for (optimizer, scheduler) in zip(optimizers, schedulers)]
        multi_scheduler = MultiScheduler(schedulers)
        return multi_optimizer, multi_scheduler
    else:
        return optimizers[0], schedulers[0]
