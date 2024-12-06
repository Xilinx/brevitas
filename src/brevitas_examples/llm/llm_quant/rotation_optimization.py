"""
Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

from dataclasses import dataclass
from dataclasses import field
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import transformers
from transformers import default_data_collator
from transformers import Trainer
from transformers.tokenization_utils import PreTrainedTokenizerBase

from brevitas.nn.equalized_layer import UnfusedRotatedModule
from brevitas.optim.sgdg import SGDG


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
        default="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",
        metadata={"help": "Huggingface access token to access gated repo like Llama"},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    output_dir: Optional[str] = field(default="/tmp/output/")
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


def apply_rotation_optimization(
        graph_model: torch.fx.GraphModule,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        unknown_args=None) -> None:
    # Get training arguments
    training_args = parse_optimization_rotation_args(unknown_args)
    # Collect trainable matrices
    trainable_parameters = []
    for module in graph_model.modules():
        if isinstance(module, UnfusedRotatedModule):
            if not module.is_sink:
                trainable_parameters.append(module.rot_mat)
    # Initialize optimizer
    optimizer = SGDG(trainable_parameters, lr=training_args.learning_rate, stiefel=True)
    trainer = Trainer(
        model=graph_model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=default_data_collator,
        optimizers=(optimizer, None))
    trainer.train()
