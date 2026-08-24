# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from dataclasses import dataclass
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
import transformers
from transformers import Trainer

try:
    from transformers.tokenization_utils import PreTrainedTokenizerBase
except:
    # This has changed in transformers v5
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrices
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
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


def _is_fsdp_enabled(training_args: transformers.TrainingArguments) -> bool:
    return (
        training_args.distributed_state.distributed_type == DistributedType.FSDP or
        os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true")


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
        extra_args: Optional[List[str]] = None,
        skip_training: bool = False,
        return_state_dict: bool = True) -> Optional[Dict[str, torch.Tensor]]:
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
    skip_training : bool
        Skip Trainer execution, used when loading an existing quantized checkpoint.
    return_state_dict : bool
        Collect and return a full CPU state dictionary after FSDP training. Non-FSDP training
        updates the supplied model in place and always returns ``None``.
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
    fsdp_enabled = _is_fsdp_enabled(training_args)
    if skip_training:
        if fsdp_enabled:
            training_args.distributed_state.destroy_process_group()
        return
    # Remove hooks and empty cache before starting training
    remove_hooks(model)
    torch.cuda.empty_cache()
    if training_args.optimizer_scheduler_args is not None:
        # Configured parameter selectors re-enable only their assigned groups.
        for param in model.parameters():
            param.requires_grad = False

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        # `tokenizer` renamed to `processing_class` in transformers 4.46, removed in 5.x.
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collate_fn)

    # Wire the teacher model whenever the selected trainer is a
    # GeneralizedTrainer subclass and distillation loss is enabled.
    if issubclass(trainer_cls, GeneralizedTrainer) and getattr(
            training_args, 'use_distillation_loss', False):
        teacher_model = copy.deepcopy(model.cpu())
        for param in teacher_model.parameters():
            param.requires_grad = False
        trainer_kwargs["teacher_model"] = teacher_model

    trainer = None
    try:
        if fsdp_enabled and not issubclass(trainer_cls, GeneralizedTrainer):
            raise RuntimeError("FSDP2 fine-tuning requires a GeneralizedTrainer subclass.")
        trainer = trainer_cls(**trainer_kwargs)
        if fsdp_enabled and not trainer.accelerator.is_fsdp2:
            raise RuntimeError("LLM distributed fine-tuning supports FSDP2 only.")
        trainer.train()
        if fsdp_enabled:
            state_dict = (
                trainer.accelerator.get_state_dict(trainer.model) if return_state_dict else None)
            trainer.accelerator.wait_for_everyone()
            return state_dict
        # After finishing training, set eval mode again
        model.eval()
        return None
    finally:
        if fsdp_enabled:
            if trainer is not None:
                trainer.accelerator.end_training()
            else:
                training_args.distributed_state.destroy_process_group()
