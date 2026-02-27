# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
import os
from typing import Any
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

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.optim.cailey_sgd import CaileySGD
from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrices
from brevitas.utils.python_utils import Registry
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks

# Registries for out-of-source customization of the training process.
# Users can register custom trainers, training argument classes, and
# optimizer/scheduler/param configurations via a plugin .py file.
TRAINER_REGISTRY = Registry[type](registry_name="TrainerRegistry")
TRAINING_ARGS_REGISTRY = Registry[type](registry_name="TrainingArgsRegistry")
OPTIMIZER_CONFIG_REGISTRY = Registry[type](registry_name="OptimizerConfigRegistry")


class MultiOptimizer:
    """Wrapper to handle multiple optimizers as a single optimizer for Trainer.

    Allows attaching different optimizer/scheduler pairs to different parameter
    groups (e.g. CaileySGD for rotation matrices and AdamW for other params).
    """

    def __init__(self, optimizers: List[torch.optim.Optimizer]) -> None:
        self.optimizers = optimizers

    def zero_grad(self, set_to_none: bool = False) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        for optimizer in self.optimizers:
            loss = optimizer.step(closure=closure)
        return loss

    @property
    def state(self) -> Dict[str, Any]:
        return {k: v for optimizer in self.optimizers for k, v in optimizer.state.items()}

    @property
    def param_groups(self) -> List[Dict[str, Any]]:
        return [
            param_group for optimizer in self.optimizers for param_group in optimizer.param_groups]


class MultiScheduler:
    """Wrapper to handle multiple schedulers as a single scheduler for Trainer.

    Schedulers in the list may be ``None`` to indicate no scheduling for the
    corresponding optimizer.
    """

    def __init__(self, schedulers: List[Optional[Any]]) -> None:
        self.schedulers = schedulers if schedulers else []

    def step(self, *args, **kwargs) -> None:
        for scheduler in self.schedulers:
            if scheduler is not None:
                scheduler.step(*args, **kwargs)

    def get_last_lr(self) -> List[float]:
        if not self.schedulers or self.schedulers[0] is None:
            return []
        return self.schedulers[0].get_last_lr()

    @property
    def state_dict(self) -> List[Optional[Dict[str, Any]]]:
        return [scheduler.state_dict() if scheduler else None for scheduler in self.schedulers]

    def load_state_dict(self, state_dicts: List[Optional[Dict[str, Any]]]) -> None:
        for scheduler, state_dict in zip(self.schedulers, state_dicts):
            if scheduler and state_dict:
                scheduler.load_state_dict(state_dict)


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


class GeneralizedTrainer(Trainer):

    def __init__(self, args: TrainingArguments = None, **kwargs) -> None:
        super().__init__(args=args, **kwargs)
        self.use_distillation_loss = args.use_distillation_loss
        self.gamma = args.gamma
        self.temperature = args.temperature

    @staticmethod
    def forward_kl_loss(
            student_logits, teacher_logits, temperature=1.0, topk=-1, reduction="batchmean"):

        if topk > 0:
            teacher_logits, indices = teacher_logits.topk(topk, dim=-1, sorted=False)
            student_log_probs = student_log_probs.gather(-1, indices).flatten(0, -2)

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
        student_log_probs = student_log_probs

        loss = F.kl_div(student_log_probs, teacher_log_probs, reduction=reduction, log_target=True)
        return loss

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
            with torch.no_grad(), quantization_status_manager(model, disable_act_quant=True, disable_weight_quant=True, disable_bias_quant=True):
                fp_outputs = model(**inputs)
            # Compute the distillation loss
            distill_loss = GeneralizedTrainer.forward_kl_loss(
                student_logits=outputs.logits,
                teacher_logits=fp_outputs.logits,
                temperature=self.temperature,
            )
            if (self.args.average_tokens_across_devices and
                (self.model_accepts_loss_kwargs or self.compute_loss_func) and
                    num_items_in_batch is not None):
                distill_loss = distill_loss * self.accelerator.num_processes
            loss = self.gamma * loss + (1. - self.gamma) * distill_loss

        return (loss, outputs) if return_outputs else loss


def parse_rotation_optimization_args(
    extra_args: Optional[List[str]] = None,
    training_args_cls: Optional[Type[transformers.TrainingArguments]] = None,
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


def _build_default_optimizers(
    model: torch.nn.Module,
    training_args: TrainingArguments,
) -> tuple:
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
    optimizer_configs: List[Dict[str, Any]],
) -> tuple:
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
            schedulers.append(None)

    multi_optimizer = MultiOptimizer(optimizers)
    multi_scheduler = MultiScheduler(schedulers) if any(s is not None for s in schedulers) else None
    return multi_optimizer, multi_scheduler


def apply_rotation_optimization(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    training_args: transformers.TrainingArguments,
    collate_fn: Callable,
    trainer_cls: Optional[Type[Trainer]] = None,
    callbacks: Optional[List[Any]] = None,
    optimizer_configs: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Optimize rotation matrices inserted into the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose rotation matrices will be optimized.
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
        (the default), the original single-optimizer behaviour is used
        (``CaileySGD`` on the rotation matrices only).
    """

    # Prepare dataset and model for training
    train_dataset = _prepare_train_dataset(train_dataset)
    model = _prepare_model(model)
    # Enable skipping optimization
    if training_args.max_steps <= 0:
        return
    # Remove hooks and empty cache before starting optimization
    remove_hooks(model)
    torch.cuda.empty_cache()
    # Freeze all model parameters; individual param groups will be
    # unfrozen by the optimizer-building helpers.
    for param in model.parameters():
        param.requires_grad = False

    # Build optimizer / scheduler pair
    if optimizer_configs is not None:
        optimizer, scheduler = _build_optimizers_from_configs(
            model, training_args, optimizer_configs)
    else:
        optimizer, scheduler = _build_default_optimizers(model, training_args)

    # Select trainer class
    if trainer_cls is None:
        trainer_cls = GeneralizedTrainer

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler))
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    # After finishing training, set eval mode again
    model.eval()
