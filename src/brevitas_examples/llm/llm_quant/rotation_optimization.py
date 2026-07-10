# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Type

from accelerate.utils import DistributedType
from datasets import Dataset
import torch
import transformers
from transformers import Trainer

try:
    from transformers.tokenization_utils import PreTrainedTokenizerBase
except:
    # This has changed in transformers v5
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrices
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
# Optimizer/scheduler building and trainer plumbing live in trainer_utils.
from brevitas_examples.llm.llm_quant.trainer_utils import _build_optimizers_from_configs
from brevitas_examples.llm.llm_quant.trainer_utils import GeneralizedTrainer
from brevitas_examples.llm.llm_quant.trainer_utils import TrainingArguments


@dataclass
class RotationTrainingArguments(TrainingArguments):
    """Training arguments for the default rotation-optimization flow.

    Expresses the CaileySGD-on-rotation-matrices default through the standard
    ``optimizer_scheduler_args`` mechanism: a single optimizer whose (single)
    parameter group is optimized with ``CaileySGD`` on the Stiefel manifold.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.optimizer_scheduler_args is None:
            self.optimizer_scheduler_args = [{
                "optimizer_cls":
                    "CaileySGD",
                "param_setup": [{
                    "get_param_fn": _select_rotation_params,
                    "optimizer_kwargs": {
                        "lr": self.learning_rate,
                        "stiefel": True,
                        "dtype": self.optimizer_dtype,},}],}]


def _select_rotation_params(
        model: torch.nn.Module,
        training_args: transformers.TrainingArguments) -> List[torch.nn.Parameter]:
    """Return the model's trainable rotation matrices (one parameter group)."""
    return extract_trainable_rotation_matrices(model)


class RotationTrainer(GeneralizedTrainer):
    """Default trainer for rotation optimization.

    Uses :class:`RotationTrainingArguments`, whose ``optimizer_scheduler_args``
    expresses CaileySGD on the trainable rotation matrices (selected via
    ``param_setup``). Selected automatically by :func:`apply_fine_tuning` when
    the model has trainable rotation matrices and no custom trainer is provided.
    """
    training_args_cls: Type[transformers.TrainingArguments] = RotationTrainingArguments


def parse_rotation_optimization_args(
    extra_args: List[str],
    trainer_cls: Type[Trainer],
    training_args_cls: Optional[Type[transformers.TrainingArguments]] = None
) -> transformers.TrainingArguments:
    """Parse *extra_args* into a training-arguments dataclass.

    The training-arguments class is resolved with the following precedence:

    1. *training_args_cls*, when explicitly provided.
    2. ``trainer_cls.training_args_cls``, when a *trainer_cls* exposing that
       attribute is provided.
    3. the built-in :class:`TrainingArguments`.
    """
    if training_args_cls is None:
        training_args_cls = getattr(trainer_cls, "training_args_cls", TrainingArguments)

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


def apply_fine_tuning(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        collate_fn: Callable,
        trainer_cls: Optional[Type[Trainer]] = None,
        extra_args: Optional[List[str]] = None) -> None:
    """Fine-tune model weights and/or rotation matrices.

    The training arguments are parsed from *extra_args* via
    :func:`parse_rotation_optimization_args`, using
    ``trainer_cls.training_args_cls`` when available. The optimizer(s) and
    scheduler(s) are built from ``training_args.optimizer_scheduler_args``. When
    that is ``None``:

    * If trainable rotation matrices are found, :class:`RotationTrainer` is used
      by default (CaileySGD on the rotations, via ``optimizer_scheduler_args``).
    * Otherwise, ``(None, None)`` is passed to the Trainer so that it uses its
      built-in optimizer (AdamW by default).

    Parameters
    ----------
    model : torch.nn.Module
        The model to fine-tune.
    tokenizer : PreTrainedTokenizerBase
        The tokenizer associated with the model.
    train_dataset : Dataset
        The training dataset.
    collate_fn : callable
        The data collator passed to the Trainer.
    trainer_cls : Type[Trainer], optional
        A custom Trainer class, typically resolved from ``TRAINER_REGISTRY``.
        Its ``training_args_cls`` class attribute customises the training
        arguments (including the optimizer/scheduler setup through
        ``optimizer_scheduler_args``). When ``None`` (the default),
        ``GeneralizedTrainer`` (or the built-in ``Trainer``) is used.
    extra_args : list of str, optional
        Raw CLI-style extra arguments parsed into the training-arguments
        dataclass (see :func:`parse_rotation_optimization_args`).
    """

    # Resolve the trainer class up front so that its ``training_args_cls`` (which
    # sets the ``optimizer_scheduler_args`` default) is used when parsing the
    # training arguments. When no custom trainer is given but the model has
    # trainable rotation matrices, default to RotationTrainer (CaileySGD on the
    # rotations, expressed through the standard optimizer_scheduler_args mechanism).
    if trainer_cls is None:
        if len(extract_trainable_rotation_matrices(model)) == 0:
            raise RuntimeError(
                "No Custom Trainer has been defined and no optimizable rotations are present in the model."
            )
        trainer_cls = RotationTrainer
    else:
        trainer_cls = trainer_cls

    # Parse the training arguments, resolving the training-args class from the
    # (possibly defaulted) trainer.
    training_args = parse_rotation_optimization_args(extra_args=extra_args, trainer_cls=trainer_cls)

    # Prepare model for training
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

    # Build optimizer / scheduler pair from the training args.
    if training_args.optimizer_scheduler_args is None:
        raise RuntimeError("TrainingArguments needs to specify optimizer_scheduler_args")

    # The optimizer-building helpers unfreeze the parameters of each
    # selected param group.
    optimizers = _build_optimizers_from_configs(model, training_args)

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collate_fn,
        optimizers=optimizers)

    # Wire the teacher model whenever the selected trainer is a
    # GeneralizedTrainer subclass and distillation loss is enabled.
    if issubclass(trainer_cls, GeneralizedTrainer) and getattr(
            training_args, 'use_distillation_loss', False):
        trainer_kwargs["teacher_model"] = copy.deepcopy(model.cpu())

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    # After finishing training, set eval mode again
    model.eval()
