from dataclasses import dataclass
from dataclasses import field
import os
from typing import List, Optional

from datasets import Dataset
import torch
import transformers
from transformers import Trainer
from transformers.tokenization_utils import PreTrainedTokenizerBase

from brevitas.optim.cailey_sgd import CaileySGD
from brevitas.utils.rotation_utils import extract_trainable_rotation_matrices
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.llm.llm_quant.data_utils import DatasetToDevice


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # By default, arguments are saved in the current working directory
    output_dir: Optional[str] = field(default=os.getcwd())


def parse_optimization_rotation_args(unknown_args=None) -> None:
    parser = transformers.HfArgumentParser(TrainingArguments)
    training_args = parser.parse_args_into_dataclasses(args=unknown_args)
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
    unknown_args: List[str] = None,
) -> None:

    # Prepare dataset and model for training
    train_dataset = _prepare_train_dataset(train_dataset)
    model = _prepare_model(model)
    # Remove hooks and empty cache before starting optimization
    remove_hooks(model)
    torch.cuda.empty_cache()
    # Get training arguments
    training_args = parse_optimization_rotation_args(unknown_args)
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
        data_collator=collate_fn,
        optimizers=(optimizer, None))
    trainer.train()
    # After finishing training, set eval mode again
    model.eval()
