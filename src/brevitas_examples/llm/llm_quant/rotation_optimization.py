# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from dataclasses import dataclass
from dataclasses import field
import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from accelerate.utils import DistributedType
from datasets import Dataset
import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer

try:
    from transformers.tokenization_utils import PreTrainedTokenizerBase
except:
    # This has changed in transformers v5
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from torch.optim.lr_scheduler import ConstantLR

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.optim.cailey_sgd import CaileySGD
from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrices
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks

# Registries for out-of-source customization of the training process.
# Users can register custom trainers, training argument classes, and
# optimizer/scheduler/param configurations via a plugin .py file.
TRAINER_REGISTRY = Registry[type](registry_name="TrainerRegistry")
TRAINING_ARGS_REGISTRY = Registry[type](registry_name="TrainingArgsRegistry")
OPTIMIZER_CONFIG_REGISTRY = Registry[type](registry_name="OptimizerConfigRegistry")


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

    ### Multi-optimizer/scheduler kwargs
    # Order-matched list of dicts.  Entry *i* supplies ``optimizer_kwargs``
    # and optionally ``scheduler_kwargs`` for the *i*-th optimizer config
    # registered via ``OPTIMIZER_CONFIG_REGISTRY``.
    optimizer_scheduler_args: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help":
                "List of dicts, each containing 'optimizer_kwargs' and optionally "
                "'scheduler_kwargs', order-matched to the optimizer configs "
                "registered via OPTIMIZER_CONFIG_REGISTRY."})

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

    def __init__(self, args: TrainingArguments = None, teacher_model=None, **kwargs) -> None:
        super().__init__(args=args, **kwargs)
        self.use_distillation_loss = args.use_distillation_loss
        self.gamma = args.gamma
        self.temperature = args.temperature
        self.kl_loss_reduction = args.kl_loss_reduction
        self.teacher_model = None if teacher_model is None else offload_model(teacher_model)

    def create_optimizer_and_scheduler(self, num_training_steps: int) -> None:
        """Build optimizer/scheduler from deferred configs when FSDP is active.

        Under FSDP the optimizer cannot be passed to the Trainer constructor
        because parameter references become stale after FSDP wraps the model.
        Instead, ``apply_fine_tuning`` stashes the optimizer configs on
        ``self.args._deferred_optimizer_configs`` and this method builds the
        optimizer after FSDP wrapping, using ``self.model`` (which now holds
        the FSDP-wrapped model with valid parameter references).
        """
        deferred = getattr(self.args, "_deferred_optimizer_configs", None)
        if deferred is None:
            return super().create_optimizer_and_scheduler(num_training_steps)

        del self.args._deferred_optimizer_configs

        if len(deferred) > 1:
            raise RuntimeError(
                "FSDP does not support MultiOptimizer (multiple optimizer "
                "groups).  Use a single optimizer config or disable FSDP.")

        os_args: Optional[List[Dict[str,
                                    Any]]] = getattr(self.args, "optimizer_scheduler_args", None)
        if os_args is None or len(os_args) < len(deferred):
            raise RuntimeError(
                "optimizer_scheduler_args on training_args does not match "
                "the deferred optimizer configs.")

        config = deferred[0]
        params = config["params"]
        if callable(params):
            params = params(self.model, self.args)
        for param in params:
            param.requires_grad = True

        entry = os_args[0]
        optimizer_class = config.get("optimizer_class", torch.optim.AdamW)
        optimizer_kwargs = entry.get("optimizer_kwargs", {})
        self.optimizer = optimizer_class(params, **optimizer_kwargs)

        scheduler_class = config.get("scheduler_class", None)
        if scheduler_class is not None:
            scheduler_kwargs = entry.get("scheduler_kwargs", {})
            self.lr_scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)
        else:
            self.create_scheduler(num_training_steps, optimizer=self.optimizer)

    @staticmethod
    def forward_kl_loss(
            student_logits, teacher_logits, temperature=1.0, topk=-1, reduction="batchmean"):
        out_dtype = student_logits.dtype
        # Apply temperature scaling
        student_logits = student_logits.float() / temperature
        teacher_logits = teacher_logits.float() / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if topk > 0:
            teacher_log_probs, indices = teacher_log_probs.topk(topk, dim=-1, sorted=False)
            student_log_probs = student_log_probs.gather(-1, indices)

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
                reduction=self.kl_loss_reduction)

            if (self.args.average_tokens_across_devices and
                (self.model_accepts_loss_kwargs or self.compute_loss_func) and
                    num_items_in_batch is not None):
                distill_loss = distill_loss * self.accelerator.num_processes
            loss = self.gamma * loss + (1. - self.gamma) * distill_loss

        return (loss, outputs) if return_outputs else loss


def parse_rotation_optimization_args(
    extra_args: Optional[List[str]] = None,
    training_args_cls: Optional[Type[transformers.TrainingArguments]] = None
) -> transformers.TrainingArguments:
    if training_args_cls is None:
        training_args_cls = TrainingArguments
    parser = transformers.HfArgumentParser(training_args_cls)
    training_args = parser.parse_args_into_dataclasses(args=extra_args)
    # If a single-process is running, only one GPU should be available
    # for Trainer, to prevent using DataParallel, which was causing an
    # error due to tensors in different devices being operated.
    # Therefore, DistributedDataParallel should be used to run in
    # multiple GPUs
    if training_args[0].distributed_state.distributed_type == DistributedType.NO and training_args[
            0]._n_gpu > 1:
        training_args[0]._n_gpu = 1
    return training_args[0]


def _prepare_model(model: torch.nn.Module) -> torch.nn.Module:
    # For a PretrainedModel, the Trainer in accelerate calls save_pretrained after
    # finishing the optimization. However, this method no longer works after
    # registering parametrizations/quantizing, so this method is mocked to prevent
    # a crash.
    def mock_save_pretrained_fn(*args, **kwargs):
        pass

    model.save_pretrained = mock_save_pretrained_fn
    # Cache needs to be disabled for training
    model.config.use_cache = False
    # Loss for training
    model.config.loss_type = "ForCausalLM"

    return model


def _prepare_train_dataset(train_dataset: Dataset) -> Dataset:
    return train_dataset


def _build_default_optimizers(model: torch.nn.Module, training_args: TrainingArguments) -> tuple:
    """Build the default (CaileySGD, None) optimizer/scheduler pair.

    Returns a tuple ``(optimizer_or_multi, scheduler_or_none)`` ready to
    be passed to the Trainer ``optimizers`` argument.
    """
    trainable_rotations = extract_trainable_rotation_matrices(model)
    for rot_mat in trainable_rotations:
        rot_mat.requires_grad = True
    optimizer = CaileySGD(
        trainable_rotations,
        lr=training_args.learning_rate,
        stiefel=True,
        dtype=training_args.optimizer_dtype)
    return optimizer, None


def _build_optimizers_from_configs(
        model: torch.nn.Module,
        training_args: transformers.TrainingArguments,
        optimizer_configs: List[Dict[str, Any]]) -> tuple:
    """Build a ``(MultiOptimizer, MultiScheduler | None)`` pair from a
    list of optimizer configuration dicts.

    Each dict in *optimizer_configs* must contain:

    * ``params`` – a list of parameters **or** a callable that receives
      ``(model, training_args)`` and returns a list of parameters.
    * ``optimizer_class`` – the optimizer class (default: ``CaileySGD``).
    * ``scheduler_class`` – an optional LR scheduler class (default:
      ``None``).

    The *optimizer_kwargs* and *scheduler_kwargs* for each entry are
    read from ``training_args.optimizer_scheduler_args[i]``, which is
    an order-matched list of dicts.  Each dict may contain:

    * ``optimizer_kwargs`` – keyword arguments forwarded to the optimizer.
    * ``scheduler_kwargs`` – keyword arguments forwarded to the scheduler.

    If ``training_args.optimizer_scheduler_args`` is ``None`` or shorter
    than *optimizer_configs*, this fails.
    """
    optimizers: List[torch.optim.Optimizer] = []
    schedulers: List[Optional[Any]] = []

    os_args: Optional[List[Dict[str,
                                Any]]] = getattr(training_args, "optimizer_scheduler_args", None)
    if os_args is None or len(os_args) < len(optimizer_configs):
        raise RuntimeError("Scheduler/Optimizer arguments do not match the configs")

    for i, config in enumerate(optimizer_configs):
        params = config["params"]
        if callable(params):
            params = params(model, training_args)
        for param in params:
            param.requires_grad = True

        # Look up kwargs from the order-matched training args list
        entry = os_args[i]

        optimizer_class = config.get("optimizer_class", CaileySGD)
        optimizer_kwargs = entry.get("optimizer_kwargs", {})
        optimizer = optimizer_class(params, **optimizer_kwargs)
        optimizers.append(optimizer)

        scheduler_class = config.get("scheduler_class", None)
        if scheduler_class is not None:
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


def apply_fine_tuning(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        training_args: transformers.TrainingArguments,
        collate_fn: Callable,
        trainer_cls: Optional[Type[Trainer]] = None,
        callbacks: Optional[List[Any]] = None,
        optimizer_configs: Optional[List[Dict[str, Any]]] = None) -> None:
    """Fine-tune model weights and/or rotation matrices.

    This is the unified training entry point.  When *optimizer_configs*
    is ``None``, the function inspects the model:

    * If trainable rotation matrices are found, a ``CaileySGD`` optimizer
      is built for them (backward-compatible rotation-optimization
      behaviour).
    * Otherwise, ``(None, None)`` is passed to the Trainer so that it
      uses its built-in optimizer (AdamW by default).

    Parameters
    ----------
    model : torch.nn.Module
        The model to fine-tune.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer associated with the model.
    train_dataset : Dataset
        The training dataset.
    training_args : transformers.TrainingArguments
        HuggingFace-compatible training arguments.  May be the built-in
        ``TrainingArguments`` or a custom subclass registered via the
        ``TRAINING_ARGS_REGISTRY``.  When *optimizer_configs* is
        provided, the ``optimizer_scheduler_args`` field on the
        training args supplies the order-matched ``optimizer_kwargs``
        and ``scheduler_kwargs`` for each entry.
    trainer_cls : Type[Trainer], optional
        A custom Trainer class.  When ``None`` (the default),
        ``GeneralizedTrainer`` is used.
    callbacks : list, optional
        A list of HuggingFace ``TrainerCallback`` instances to attach to
        the trainer.
    optimizer_configs : list of dict, optional
        A list of optimizer configuration dicts.  Each dict may contain:

        * ``params`` – a list of ``torch.nn.Parameter`` **or** a
          callable ``(model, training_args) -> List[Parameter]``.
        * ``optimizer_class`` – optimizer class (default ``CaileySGD``).
        * ``scheduler_class`` – optional LR scheduler class.

        The ``optimizer_kwargs`` and ``scheduler_kwargs`` for each
        config are read from
        ``training_args.optimizer_scheduler_args[i]``.

        When multiple configs are provided, a ``MultiOptimizer`` /
        ``MultiScheduler`` is built automatically.  When ``None``
        (the default), the behaviour depends on whether the model
        contains trainable rotation matrices (see above).
    """

    # Prepare dataset and model for training
    train_dataset = _prepare_train_dataset(train_dataset)
    model = _prepare_model(model)
    # Enable skipping training
    if training_args.max_steps <= 0:
        return
    # Remove hooks and empty cache before starting training
    remove_hooks(model)
    torch.cuda.empty_cache()
    # Freeze all model parameters; individual param groups will be
    # unfrozen by the optimizer-building helpers.
    for param in model.parameters():
        param.requires_grad = False

    # Build optimizer / scheduler pair
    if optimizer_configs is not None:
        optimizers = _build_optimizers_from_configs(model, training_args, optimizer_configs)
    elif extract_trainable_rotation_matrices(model):
        # Backward-compatible default: CaileySGD on rotation matrices
        optimizers = _build_default_optimizers(model, training_args)
    else:
        # No custom configs and no rotation matrices — let the HF
        # Trainer use its built-in optimizer.
        optimizers = (None, None)

    # Select trainer class
    if trainer_cls is None:
        trainer_cls = GeneralizedTrainer

    teacher_model = copy.deepcopy(model.cpu()) if training_args.use_distillation_loss else None

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collate_fn,
        optimizers=optimizers)
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    # After finishing training, set eval mode again
    model.eval()


# Backward-compatible alias
apply_rotation_optimization = apply_fine_tuning
