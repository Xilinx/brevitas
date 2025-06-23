# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import json
from typing import Dict, List, Optional, OrderedDict, Type, Union

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
from brevitas_examples.common.learned_round.learned_round_method import BlockLoss
from brevitas_examples.common.learned_round.learned_round_parser import \
    parse_learned_round_loss_class
from brevitas_examples.common.learned_round.learned_round_parser import parse_lr_scheduler_class
from brevitas_examples.common.learned_round.learned_round_parser import parse_optimizer_class


class TargetParametrizations(Enum):
    SCALES = "scales"
    LEARNED_ROUND = "learned_round"


# TODO: Decide whether it is worth grouping the get_target_parameters under a class
def get_round_parameters(module: nn.Module, state_dict: OrderedDict, prefix: str = "") -> bool:
    if isinstance(module, LearnedRoundSte):
        for param_name, param in module.named_parameters():
            state_dict[f"{prefix}.{param_name}"] = param
        # Early stoppping
        return True
    return False


def get_scale_parameters(module: nn.Module, state_dict: OrderedDict, prefix: str = "") -> bool:
    if isinstance(module, WeightQuantProxyFromInjectorBase):
        for param_name, param in module.named_parameters():
            if param_name.endswith('scaling_impl.value'):
                state_dict[f"{prefix}.{param_name}"] = param
        # Early stoppping
        return True
    return False


TARGET_PARAMETRIZATIONS_MAP = {
    TargetParametrizations.SCALES: get_scale_parameters,
    TargetParametrizations.LEARNED_ROUND: get_round_parameters,}


#TODO: Add license from Nanotron
def _convert_str_dict(passed_value: Dict) -> Dict:
    "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
    for key, value in passed_value.items():
        if isinstance(value, dict):
            passed_value[key] = _convert_str_dict(value)
        elif isinstance(value, str):
            # First check for bool and convert
            if value.lower() in ("true", "false"):
                passed_value[key] = value.lower() == "true"
            # Check for digit
            elif value.isdigit():
                passed_value[key] = int(value)
            elif value.replace(".", "", 1).isdigit():
                passed_value[key] = float(value)

    return passed_value


def _parse_dataclass_dict(data_cls, dict_attributes: List[str]) -> Dict:
    for attr in dict_attributes:
        if not hasattr(data_cls, attr):
            raise ValueError(f"Dataclass {type(data_cls).__name__} has no attribute named {attr}")
        kwargs = getattr(data_cls, attr)

        if kwargs is None:
            kwargs = {}
        elif isinstance(kwargs, str):
            # Parse in args that could be `dict` sent in from the CLI as a string
            kwargs = json.loads(kwargs)
            # Convert str values to types if applicable
            kwargs = _convert_str_dict(kwargs)
        elif isinstance(kwargs, dict):
            pass
        else:
            # Raise an error if the attribute cannot be parsed into a dictionary
            raise ValueError(
                f"Value set for attribute {attr} of dataclass {type(data_cls).__name__} cannot be converted into a dictionary."
            )
        # Set the updated value
        setattr(data_cls, attr, kwargs)


@dataclass
class LRSchedulerArgs:
    lr_scheduler_cls: Union[str, Type] = field(
        default="linear",
        metadata={"help": "The learning rate scheduler to use."},
    )
    lr_scheduler_kwargs: Optional[Union[Dict, str]] = field(
        default=None,
        metadata={"help": ("Extra keyword arguments for the learning rate "
                           "scheduler.")},
    )
    _DICT_ATTRIBUTES = ["lr_scheduler_kwargs"]

    def __post_init__(self) -> None:
        # Parse in args that could be `dict` sent in from the CLI as a string
        _parse_dataclass_dict(self, self._DICT_ATTRIBUTES)
        # Parse string to learning rate scheduler class if needed
        self.lr_scheduler_cls = (
            parse_lr_scheduler_class(self.lr_scheduler_cls) if isinstance(
                self.lr_scheduler_cls, str) else self.lr_scheduler_cls)


@dataclass
class OptimizerArgs:
    optimizer_cls: Union[str, Type[Optimizer]] = field(
        default="adam",
        metadata={"help": "The optimizer to use."},
    )
    lr: float = field(
        default=1e-3,
        metadata={"help": "Initial learning rate for the optimizer."},
    )
    optimizer_kwargs: Optional[Union[Dict, str]] = field(
        default=None,
        metadata={"help": "Extra keyword arguments for the optimizer."},
    )
    lr_scheduler_args: Optional[LRSchedulerArgs] = field(
        default=None,
        metadata={
            "help": ("Hyperparameters of learning rate scheduler for the selected"
                     "optimizer.")},
    )
    _DICT_ATTRIBUTES = ["optimizer_kwargs"]

    def __post_init__(self) -> None:
        # Parse in args that could be `dict` sent in from the CLI as a string
        _parse_dataclass_dict(self, self._DICT_ATTRIBUTES)
        # Parse optimizer name to class
        self.optimizer_cls = (
            parse_optimizer_class(self.optimizer_cls)
            if isinstance(self.optimizer_cls, str) else self.optimizer_cls)
        if self.lr < 0:
            raise ValueError(f"Expected a positive learning rate but {self.lr} was passed.")


@dataclass
class TrainingArgs:
    optimizers_args: List[OptimizerArgs] = field(
        metadata={"help": ("Hyperparameters of the optimizers to use during training.")})
    optimizers_targets: List[Union[str, TargetParametrizations]] = field(
        metadata={"help": ("Targets to be optimized.")})
    block_name_attribute: str = field(
        metadata={"help": ("Attribute with the blocks to be optimized.")})
    batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for training."})
    iters: int = field(default=200, metadata={"help": "Number of training iterations."})
    loss_scaling_factor: float = field(
        default=1000.,
        metadata={"help": "Scaling factor for the loss."},
    )
    use_best_model: bool = field(
        default=True,
        metadata={
            "help":
                ("Whether to use the best setting of the learned round found "
                 "during training.")})
    use_amp: bool = field(
        default=True,
        metadata={"help": "Whether to train using PyTorch Automatic Mixed Precision."})
    amp_dtype: Union[str, torch.dtype] = field(
        default=torch.float16,
        metadata={
            "choices": ["float16", "bfloat16"], "help": "Dtype for mixed-precision training."})

    def __post_init__(self) -> None:
        for optimizer_args in self.optimizers_args:
            # Check if the optimizer has an attached learning rate scheduler
            if optimizer_args.lr_scheduler_args is not None:
                optimizer_args.lr_scheduler_args.lr_scheduler_kwargs["total_iters"] = self.iters
        # Initialize the target parametrizations
        self.optimizers_targets = [
            TargetParametrizations(optimizer_target)
            if isinstance(optimizer_target, str) else optimizer_target
            for optimizer_target in self.optimizers_targets]
        # Parse amp_dtype
        self.amp_dtype = getattr(torch, self.amp_dtype) if isinstance(
            self.amp_dtype, str) else self.amp_dtype


@dataclass
class LearnedRoundArgs:
    learned_round_param: Union[str, LearnedRoundImplType] = field(
        default="identity",
        metadata={
            "help": "Defines the functional form of the learned round parametrization.",
            "choices": [param.value.lower() for param in LearnedRoundImplType]})
    learned_round_kwargs: Optional[Union[Dict, str]] = field(
        default=None,
        metadata={"help": "Extra keyword arguments for the learned round parametrization."},
    )
    loss_cls: Union[str, Type[BlockLoss]] = field(
        default="mse", metadata={"help": "Class of the loss to be used for rounding optimization."})
    loss_kwargs: Optional[Union[Dict, str]] = field(
        default=None,
        metadata={"help": "Extra keyword arguments for the learned round loss."},
    )
    fast_update: bool = field(
        default=True, metadata={"help": ("Whether to use fast update with learned round.")})

    _DICT_ATTRIBUTES = ["learned_round_kwargs", "loss_kwargs"]

    def __post_init__(self) -> None:
        # Parse in args that could be `dict` sent in from the CLI as a string
        _parse_dataclass_dict(self, self._DICT_ATTRIBUTES)

        self.learned_round_param = LearnedRoundImplType(
            self.learned_round_param.upper()) if isinstance(
                self.learned_round_param, str) else self.learned_round_param

        self.loss_cls = (
            parse_learned_round_loss_class(self.loss_cls)
            if isinstance(self.loss_cls, str) else self.loss_cls)


@dataclass
class Config:
    learned_round_args: LearnedRoundArgs = field(
        metadata={"help": "Learned round parametrization."})
    training_args: TrainingArgs = field(metadata={"help": "Hyperparameters for optimization."})
