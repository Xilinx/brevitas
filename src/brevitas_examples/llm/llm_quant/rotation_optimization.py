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
import torch.nn.functional as F
import transformers
from transformers import get_scheduler
from transformers import Trainer

try:
    from transformers.tokenization_utils import PreTrainedTokenizerBase
except:
    # This has changed in transformers v5
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrices
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
# Optimizer/scheduler building and trainer plumbing live in trainer_utils.
from brevitas_examples.llm.llm_quant.trainer_utils import _build_optimizers_from_configs
from brevitas_examples.llm.llm_quant.trainer_utils import MultiOptimizer
from brevitas_examples.llm.llm_quant.trainer_utils import MultiScheduler
from brevitas_examples.llm.llm_quant.trainer_utils import TrainingArguments


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
        student_logits = student_logits.float() / temperature
        teacher_logits = teacher_logits.float() / temperature

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
    extra_args: Optional[List[str]] = None,
    training_args_cls: Optional[Type[transformers.TrainingArguments]] = None,
    trainer_cls: Optional[Type[Trainer]] = None,
) -> transformers.TrainingArguments:
    """Parse *extra_args* into a training-arguments dataclass.

    The training-arguments class is resolved with the following precedence:

    1. *training_args_cls*, when explicitly provided.
    2. ``trainer_cls.training_args_cls``, when a *trainer_cls* exposing that
       attribute is provided.
    3. the built-in :class:`TrainingArguments`.
    """
    if training_args_cls is None and trainer_cls is not None:
        training_args_cls = getattr(trainer_cls, "training_args_cls", None)
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


def apply_fine_tuning(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        collate_fn: Callable,
        custom_trainer_cls: Optional[Type[Trainer]] = None,
        extra_args: Optional[List[str]] = None) -> None:
    """Fine-tune model weights and/or rotation matrices.

    The training arguments are parsed from *extra_args* via
    :func:`parse_rotation_optimization_args`, using
    ``custom_trainer_cls.training_args_cls`` when available. The optimizer(s) and
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
    custom_trainer_cls : Type[Trainer], optional
        A custom Trainer class, typically resolved from ``TRAINER_REGISTRY``.
        Its ``training_args_cls`` class attribute customises the training
        arguments (including the optimizer/scheduler setup through
        ``optimizer_scheduler_args``). When ``None`` (the default),
        ``GeneralizedTrainer`` (or the built-in ``Trainer``) is used.
    extra_args : list of str, optional
        Raw CLI-style extra arguments parsed into the training-arguments
        dataclass (see :func:`parse_rotation_optimization_args`).
    """

    # Parse the training arguments, resolving the training-args class from the
    # (possibly defaulted) trainer.
    training_args = parse_rotation_optimization_args(
        extra_args=extra_args, trainer_cls=custom_trainer_cls)

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

    # When no custom trainer is given but the model has trainable rotation
    # matrices, default to RotationTrainer (CaileySGD on the rotations, expressed
    # through the standard optimizer_scheduler_args mechanism).
    if custom_trainer_cls is None:
        if len(extract_trainable_rotation_matrices(model)) == 0:
            raise RuntimeError(
                "No Custom Trainer has been defined and no optimizable rotations are present in the model."
            )
        trainer_cls = RotationTrainer
    else:
        trainer_cls = custom_trainer_cls

    # Wire the teacher model whenever the selected trainer is a
    # GeneralizedTrainer subclass and distillation loss is enabled.
    if issubclass(trainer_cls, GeneralizedTrainer) and getattr(
            training_args, 'use_distillation_loss', False):
        trainer_kwargs["teacher_model"] = copy.deepcopy(model.cpu())

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    # After finishing training, set eval mode again
    model.eval()
