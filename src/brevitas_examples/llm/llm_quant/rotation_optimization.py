# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
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

from brevitas.utils.parametrization_utils import cast_parameters_
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

    rotation_parameter_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help":
                "Storage dtype for the trainable rotation matrices (e.g. 'float32'). "
                "When set, rotation masters are kept in this dtype while the model and "
                "the rotation forward remain in the model dtype. None keeps the model "
                "dtype."})

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


def _resolve_dtype(name: Optional[str], option: str) -> Optional[torch.dtype]:
    """Resolve a dtype name (e.g. ``'float32'``) into a ``torch.dtype``."""
    if name is None:
        return None
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{option} must name a torch dtype, got {name!r}")
    return dtype


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

    @classmethod
    def prepare_model_for_training(
            cls, model: torch.nn.Module, args: RotationTrainingArguments) -> None:
        """Set the requested storage dtype for trainable rotation matrices.

        Runs before quant-proxy compilation so the parameter dtype is fixed
        before compilation captures it. Idempotent: re-running with the same
        ``args`` is a no-op once the dtype already matches.
        """
        cast_parameters_(
            extract_trainable_rotation_matrices(model),
            _resolve_dtype(args.rotation_parameter_dtype, "rotation_parameter_dtype"))


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


def _resolve_trainer_cls(model: torch.nn.Module,
                         trainer_cls: Optional[Type[Trainer]]) -> Type[Trainer]:
    """Resolve the trainer class used to prepare and optimize ``model``.

    When no custom trainer is provided, default to :class:`RotationTrainer` if
    the model has trainable rotation matrices, otherwise raise.
    """
    if trainer_cls is None:
        if len(extract_trainable_rotation_matrices(model)) == 0:
            raise RuntimeError(
                "No Custom Trainer has been defined and no optimizable rotations are present in the model."
            )
        return RotationTrainer
    return trainer_cls


def _maybe_prepare_model_for_training(
        trainer_cls: Type[Trainer],
        model: torch.nn.Module,
        training_args: transformers.TrainingArguments) -> None:
    """Run the trainer's optional, idempotent pre-compilation hook, if any."""
    prepare = getattr(trainer_cls, "prepare_model_for_training", None)
    if prepare is not None:
        prepare(model, training_args)


def prepare_fine_tuning(
        model: torch.nn.Module,
        trainer_cls: Optional[Type[Trainer]] = None,
        extra_args: Optional[List[str]] = None
) -> Tuple[Type[Trainer], transformers.TrainingArguments]:
    """Resolve the trainer/training-args and prepare the model before compile.

    This is meant to be called after quantization/post-processing but *before*
    quant-proxy compilation and the first calibration forward, so a trainer can
    establish parameter sharing or storage dtypes before compilation captures
    parameter identities/dtypes. It returns the resolved ``(trainer_cls,
    training_args)`` so the caller can pass them straight to
    :func:`apply_fine_tuning` without re-parsing.
    """
    trainer_cls = _resolve_trainer_cls(model, trainer_cls)
    training_args = parse_rotation_optimization_args(
        extra_args=extra_args if extra_args is not None else [], trainer_cls=trainer_cls)
    _maybe_prepare_model_for_training(trainer_cls, model, training_args)
    return trainer_cls, training_args


def apply_fine_tuning(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        collate_fn: Callable,
        trainer_cls: Optional[Type[Trainer]] = None,
        extra_args: Optional[List[str]] = None,
        training_args: Optional[transformers.TrainingArguments] = None) -> None:
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
    training_args : transformers.TrainingArguments, optional
        Pre-resolved training arguments, as returned by
        :func:`prepare_fine_tuning`. When provided, *extra_args* is ignored and
        the model-preparation hook is not re-run (it already ran in
        :func:`prepare_fine_tuning`). When ``None``, the trainer is resolved and
        the model is prepared here.
    """

    # Resolve the trainer class and training arguments. When training_args is
    # not supplied, this also runs the (idempotent) pre-training model
    # preparation hook so that direct callers get the same behaviour as the LLM
    # entrypoint, which prepares the model before quant-proxy compilation.
    if training_args is None:
        trainer_cls, training_args = prepare_fine_tuning(
            model=model, trainer_cls=trainer_cls, extra_args=extra_args)
    else:
        trainer_cls = _resolve_trainer_cls(model, trainer_cls)

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
        # `tokenizer` renamed to `processing_class` in transformers 4.46, removed in 5.x.
        processing_class=tokenizer,
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
