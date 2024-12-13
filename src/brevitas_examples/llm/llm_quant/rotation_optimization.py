"""
Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.tokenization_utils import PreTrainedTokenizerBase

from brevitas.optim.sgdg import SGDG
from brevitas_examples.llm.llm_quant.rotation_utils import extract_trainable_rotation_matrices


@dataclass
class ModelArguments:
    input_model: Optional[str] = field(
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        metadata={"help": "Input model"})
    output_rotation_path: Optional[str] = field(
        default="test-output", metadata={"help": "Output rotation checkpoint path"})
    optimized_rotation_path: Optional[str] = field(
        default=None, metadata={"help": "Optimized rotation checkpoint path"})
    access_token: Optional[str] = field(
        default="",
        metadata={"help": "Huggingface access token to access gated repo like Llama"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="/tmp/output/")
    use_cpu: Optional[bool] = field(default="False")
    model_max_length: Optional[int] = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)"},
    )


def parse_optimization_rotation_args(unknown_args=None) -> None:
    parser = transformers.HfArgumentParser((
        ModelArguments,
        TrainingArguments,
    ))
    _, training_args = parser.parse_args_into_dataclasses(args=unknown_args)
    return training_args


def collate_fn(kwargs_list, return_tensors="pt"):
    # Keyword arguments
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
    # FP outputs
    return kwargs


def apply_rotation_optimization(
        graph_model: torch.fx.GraphModule,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        unknown_args=None) -> None:
    # Get training arguments
    training_args = parse_optimization_rotation_args(unknown_args)
    # Set to False the model parameters
    for param in graph_model.parameters():
        param.requires_grad = False
    # Collect trainable matrices
    trainable_rotations = extract_trainable_rotation_matrices(graph_model)
    for rot_mat in trainable_rotations:
        rot_mat.requires_grad = True
    # Initialize optimizer
    optimizer = SGDG(trainable_rotations, lr=training_args.learning_rate, stiefel=True)
    trainer = Trainer(
        model=graph_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collate_fn,
        optimizers=(optimizer, None))
    trainer.train()
