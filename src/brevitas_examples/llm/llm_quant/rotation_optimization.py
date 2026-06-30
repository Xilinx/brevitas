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
from typing import Tuple
from typing import Type

from accelerate.utils import DistributedType
from datasets import Dataset
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
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
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
# Optimizer/scheduler building and trainer plumbing live in trainer_utils.
from brevitas_examples.llm.llm_quant.trainer_utils import _build_optimizers_from_configs
from brevitas_examples.llm.llm_quant.trainer_utils import OptimizerParamsSpec


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
    # Order-matched list of dicts.  Entry *i* fully describes the *i*-th
    # optimizer (and its optional scheduler) for the *i*-th entry of the
    # trainer's ``optimizer_setup``.  Each dict may contain:
    #   * 'optimizer_cls'    : optimizer class *name* (str), resolved against
    #                          the optimizer namespaces. Defaults to CaileySGD.
    #   * 'optimizer_kwargs' : a list of dicts, one per parameter group of the
    #                          matching optimizer (always a list, even for a
    #                          single group; a bare dict is rejected).
    #   * 'scheduler_cls'    : optional LR scheduler class *name* (str).
    #   * 'scheduler_kwargs' : optional dict of kwargs for the scheduler.
    optimizer_scheduler_args: Optional[List[Dict[str, Any]]] = field(
        default=None,
        metadata={
            "help":
                "List of dicts describing each optimizer/scheduler, order-matched "
                "to the trainer's optimizer_setup entries. Each dict may contain "
                "'optimizer_cls' (str), 'optimizer_kwargs' (list of dicts, one "
                "per parameter group), 'scheduler_cls' (str) and "
                "'scheduler_kwargs' (dict)."})

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

    # Class attributes consumed by the LLM entrypoint when this trainer is
    # registered via ``--custom-trainer``. Subclasses may override them to
    # customise the training arguments and optimizer/scheduler setup. When left
    # at their defaults (``training_args_cls`` is the built-in
    # ``TrainingArguments`` and ``optimizer_setup`` is ``None``), the default
    # behaviour of the LLM example is used.
    training_args_cls: Type[transformers.TrainingArguments] = TrainingArguments
    optimizer_setup: Optional[Callable[[], List[OptimizerParamsSpec]]] = None

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


def _build_rotation_optimizers(
    model: torch.nn.Module,
    training_args: TrainingArguments,
) -> Tuple[Optimizer, Optional[LRScheduler]]:
    """Build the default (CaileySGD, None) optimizer/scheduler pair.

    Returns a tuple ``(optimizer, scheduler_or_none)`` ready to be passed to the
    Trainer ``optimizers`` argument.
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


def apply_fine_tuning(
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        collate_fn: Callable,
        custom_trainer_cls: Optional[Type[Trainer]] = None,
        extra_args: Optional[List[str]] = None,
        callbacks: Optional[List[Any]] = None) -> None:
    """Fine-tune model weights and/or rotation matrices.

    The training arguments are parsed from *extra_args* via
    :func:`parse_rotation_optimization_args`, using
    ``custom_trainer_cls.training_args_cls`` when available. The optimizer setup
    is read from ``custom_trainer_cls.optimizer_setup``. When that setup is
    ``None``, the function inspects the model:

    * If trainable rotation matrices are found, a ``CaileySGD`` optimizer
      is built for them (the default rotation-optimization behaviour).
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
    collate_fn : callable
        The data collator passed to the Trainer.
    custom_trainer_cls : Type[Trainer], optional
        A custom Trainer class, typically resolved from ``TRAINER_REGISTRY``.
        Its ``training_args_cls`` and ``optimizer_setup`` class attributes
        customise the training arguments and optimizer setup. When ``None``
        (the default), ``GeneralizedTrainer`` (or the built-in ``Trainer``) is
        used and the optimizer setup defaults apply.
    extra_args : list of str, optional
        Raw CLI-style extra arguments parsed into the training-arguments
        dataclass (see :func:`parse_rotation_optimization_args`).
    callbacks : list, optional
        A list of HuggingFace ``TrainerCallback`` instances to attach to
        the trainer.
    """
    # Parse the training arguments, resolving the training-args class from the
    # custom trainer when provided.
    training_args = parse_rotation_optimization_args(
        extra_args=extra_args, trainer_cls=custom_trainer_cls)

    # Read the optimizer setup from the custom trainer (if any). It may be a
    # callable returning the list (deferred construction) or the list directly.
    optimizer_setup = getattr(custom_trainer_cls, "optimizer_setup", None)
    if callable(optimizer_setup):
        optimizer_setup = optimizer_setup()

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

    # Build optimizer / scheduler pair
    if optimizer_setup is not None:
        optimizers = _build_optimizers_from_configs(model, training_args, optimizer_setup)
    elif extract_trainable_rotation_matrices(model):
        # Default when no configs are given but rotations are present:
        # CaileySGD on the rotation matrices.
        optimizers = _build_rotation_optimizers(model, training_args)
    else:
        # No custom configs and no rotation matrices — let the HF
        # Trainer use its built-in optimizer.
        optimizers = (None, None)

    trainer_kwargs: Dict[str, Any] = dict(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collate_fn,
        optimizers=optimizers)

    # Select trainer class
    trainer_cls = custom_trainer_cls
    if trainer_cls is None:
        if getattr(training_args, 'use_distillation_loss', False):
            trainer_cls = GeneralizedTrainer
            teacher_model = copy.deepcopy(model.cpu())
            trainer_kwargs["teacher_model"] = teacher_model
        else:
            trainer_cls = Trainer

    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks

    trainer = trainer_cls(**trainer_kwargs)
    trainer.train()
    # After finishing training, set eval mode again
    model.eval()
