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
import transformers
from transformers import Trainer
from transformers.data.data_collator import InputDataClass
from transformers.tokenization_utils import PreTrainedTokenizerBase

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
    trainer = Trainer(
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
