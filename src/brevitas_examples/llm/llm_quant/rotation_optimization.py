# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from accelerate.utils import DistributedType
from datasets import Dataset
import torch
import torch.nn.functional as F
import transformers
from transformers import Trainer
from transformers.data.data_collator import InputDataClass
from transformers.tokenization_utils import PreTrainedTokenizerBase

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.optim.cailey_sgd import CaileySGD
from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrices
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.llm.llm_quant.data_utils import collate_fn
from brevitas_examples.llm.llm_quant.data_utils import DatasetToDevice


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # By default, arguments are saved in the current working directory
    output_dir: Optional[str] = field(default=os.getcwd())
    # NOTE: Currently, there is no infrastructure to resume training
    # from a checkpoint, so related files are not save by default
    save_strategy: Optional[str] = field(default="no")

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
def data_collator(kwargs_list: List[InputDataClass], return_tensors: str = "pt") -> Dict[str, Any]:
    assert (return_tensors == "pt") or (return_tensors is None), f"Only 'pt' is supported as a value for return_tensors. However {return_tensors} was received."
    return collate_fn(kwargs_list)


def _prepare_train_dataset(train_dataset: DatasetToDevice) -> Dataset:
    return DatasetToDevice(
        data=[
            {
                # setting "labels" to train_datapoint["input_ids"] is correct since "labels"
                # are just input_ids shifted by 1 and this shift is handled later on.
                "input_ids": train_datapoint["input_ids"],
                "labels": train_datapoint["input_ids"]} for train_datapoint in train_dataset.data],
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
        data_collator=data_collator,
        optimizers=(optimizer, None))
    trainer.train()
    # After finishing training, set eval mode again
    model.eval()
