# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
import os
from typing import List
from typing import Optional

from accelerate.utils import DistributedType
from datasets import Dataset
import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer
from transformers.tokenization_utils import PreTrainedTokenizerBase

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.optim.cailey_sgd import CaileySGD
from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrices
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.llm.llm_quant.data_utils import DatasetToDevice


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # By default, arguments are saved in the current working directory
    output_dir: Optional[str] = field(default=os.getcwd())
    # NOTE: Currently, there is no infrastructure to resume training
    # from a checkpoint, so related files are not save by default
    save_strategy: Optional[str] = field(default="no")

    ### Distillation Loss args
    # Whether to compute the distillation loss
    use_distillation_loss: bool = field(default=False)
    # Weight given to the CE loss term in the overall loss
    # The distillation terms is given a weight of 1 - \gamma
    gamma: float = field(default=0.1)
    # Softmax temperature for the soft targets
    temperature: float = field(default=1.0)
    # Interpolation coefficient between 0 and 1, in the generalized
    # JS divergence
    beta: float = field(default=0.5)


class GeneralizedTrainer(Trainer):

    def __init__(self, args: TrainingArguments = None, **kwargs) -> None:
        super().__init__(args=args, **kwargs)
        self.use_distillation_loss = args.use_distillation_loss
        self.gamma = args.gamma
        self.temperature = args.temperature
        self.beta = args.beta

    @staticmethod
    def generalized_jsd_loss(
            student_logits,
            teacher_logits,
            labels=None,
            beta=0.5,
            temperature=1.0,
            reduction="batchmean"):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits: Tensor of shape (batch_size, sequence_length, vocab_size)
            labels: Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing loss
            beta: Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature: Softmax temperature (default: 1.0)
            reduction: Specifies the reduction to apply to the output (default: 'batchmean')

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        # Apply temperature scaling
        student_logits = student_logits / temperature
        teacher_logits = teacher_logits / temperature

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        # Compute the log of the mixture distribution
        # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
        beta = torch.tensor(beta, dtype=student_log_probs.dtype)
        mixture_log_probs = torch.logsumexp(
            torch.stack([
                student_log_probs + torch.log(beta), teacher_log_probs + torch.log(1 - beta)]),
            dim=0,
        )

        # Compute KL divergences using F.kl_div
        # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
        kl_teacher = F.kl_div(
            mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
        kl_student = F.kl_div(
            mixture_log_probs, student_log_probs, reduction="none", log_target=True)

        # Compute the Generalized Jensen-Shannon Divergence
        jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (
                jsd.size(0) * jsd.size(1))
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # If distillation loss is used, we need to retrieve the original model's outputs
        return_outputs = return_outputs if not self.use_distillation_loss else True

        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        if return_outputs:
            loss, outputs = loss

        if self.use_distillation_loss:
            with torch.no_grad(), quantization_status_manager(model, disable_act_quant=True, disable_weight_quant=True, disable_bias_quant=True):
                fp_ouputs = model(**inputs)
            # Compute the distillation loss
            distill_loss = GeneralizedTrainer.generalized_jsd_loss(
                student_logits=outputs.logits,
                teacher_logits=fp_ouputs.logits,
                beta=self.beta,
                temperature=self.temperature,
            )
            loss = self.gamma * loss + (1. - self.gamma) * distill_loss

        if (self.args.average_tokens_across_devices and
            (self.model_accepts_loss_kwargs or self.compute_loss_func) and
                num_items_in_batch is not None):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


def parse_rotation_optimization_args(extra_args: Optional[List[str]] = None) -> TrainingArguments:
    parser = transformers.HfArgumentParser(TrainingArguments)
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


# Function to create a batch
def collate_fn(kwargs_list, return_tensors="pt"):
    kwargs = {}
    for curr_dict in kwargs_list:
        for key, value in curr_dict.items():
            if isinstance(value, torch.Tensor):
                if key not in kwargs:
                    kwargs[key] = []
                kwargs[key].append(value)
            else:
                if key not in kwargs:
                    kwargs[key] = value
    for key, value in kwargs.items():
        if isinstance(value, list) and len(value) > 0:
            kwargs[key] = torch.cat(kwargs[key], dim=0)
    return kwargs


def _prepare_train_dataset(train_dataset: DatasetToDevice) -> Dataset:
    return DatasetToDevice(
        data=[{
            "input_ids": train_datapoint["input_ids"], "labels": train_datapoint["input_ids"]}
              for train_datapoint in train_dataset.data],
        device=None)


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


def apply_rotation_optimization(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: DatasetToDevice,
    training_args: TrainingArguments,
) -> None:

    # Prepare dataset and model for training
    train_dataset = _prepare_train_dataset(train_dataset)
    model = _prepare_model(model)
    # Enable skipping optimization
    if training_args.max_steps <= 0:
        return
    # Remove hooks and empty cache before starting optimization
    remove_hooks(model)
    torch.cuda.empty_cache()
    # Set to False the model parameters
    for param in model.parameters():
        param.requires_grad = False
    # Collect trainable matrices
    trainable_rotations = extract_trainable_rotation_matrices(model)
    for rot_mat in trainable_rotations:
        rot_mat.requires_grad = True
    optimizer = CaileySGD(trainable_rotations, lr=training_args.learning_rate, stiefel=True)
    trainer = GeneralizedTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collate_fn,
        optimizers=(optimizer, None))
    trainer.train()
    # After finishing training, set eval mode again
    model.eval()
