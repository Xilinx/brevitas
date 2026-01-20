# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union
import warnings

import torch
from torch.optim.optimizer import Optimizer

from brevitas import optim
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.utils.python_utils import parse_dataclass_dicts
from brevitas_examples.common.learned_round.learned_round_method import BLOCK_LOSS_REGISTRY
from brevitas_examples.common.learned_round.learned_round_method import BlockLoss
from brevitas_examples.common.learned_round.learned_round_method import TARGET_PARAM_FN_REGISTRY
from brevitas_examples.common.learned_round.learned_round_method import TargetParamFn

OPTIMIZER_NAMESPACES = [torch.optim, optim]
LR_SCHEDULER_NAMESPACES = [torch.optim.lr_scheduler]


def _parse_optimizer_class(optimizer_str: str) -> Type[Optimizer]:
    optimizer_namespace_keys = []
    for namespace in OPTIMIZER_NAMESPACES:
        optimizer_namespace_keys += [
            (namespace, optimizer_key)
            for optimizer_key in namespace.__dict__.keys()
            # Make sure that only valid optimizer implementations are
            # retrieved, when matching the string passed by the user
            if (
                # Verify that the key stars with the one passed by the user
                optimizer_key.lower() == optimizer_str.lower() and
                # Verify that key corresponds to a class
                isinstance(namespace.__dict__[optimizer_key], type) and
                # Make sure the abstract class is not used
                optimizer_key != "Optimizer" and
                # An optimizer implements zero_grad and step. Check that this
                # is the case for the class retrieved from torch.optim
                hasattr(namespace.__dict__[optimizer_key], 'step') and
                callable(namespace.__dict__[optimizer_key].step) and
                hasattr(namespace.__dict__[optimizer_key], 'zero_grad') and
                callable(namespace.__dict__[optimizer_key].zero_grad))]

    if len(optimizer_namespace_keys) == 0:
        raise ValueError(
            f"{optimizer_str} is not a valid optimizer in namespaces {[_namespace.__name__ for _namespace in OPTIMIZER_NAMESPACES]}."
        )

    namespace, optimizer_name = optimizer_namespace_keys[0]
    if len(optimizer_namespace_keys) > 1:
        warnings.warn(
            f"There are multiple potential matches for optimizer {optimizer_str} ({[_optimizer_name for _, _optimizer_name in optimizer_namespace_keys]}). "
            f"Defaulting to {optimizer_name} from {namespace.__name__}.")

    optimizer_class = getattr(namespace, optimizer_name)
    return optimizer_class


def _parse_lr_scheduler_class(lr_scheduler_str: str) -> Type:
    lr_scheduler_namespace_keys = []
    for namespace in LR_SCHEDULER_NAMESPACES:
        lr_scheduler_namespace_keys += [
            (namespace, lr_scheduler_key)
            for lr_scheduler_key in torch.optim.lr_scheduler.__dict__.keys()
            # Check for making sure that only valid LRScheduler implementations are
            # retrived, when matching with the string passed by the user
            if
            ((
                lr_scheduler_key.lower() == lr_scheduler_str.lower() or
                lr_scheduler_key.lower() == lr_scheduler_str.lower() + "lr") and
             # Verify that key corresponds to a class
             isinstance(torch.optim.lr_scheduler.__dict__[lr_scheduler_key], type) and
             # Make sure the abstract class is not retrieved
             lr_scheduler_key != "LRScheduler" and
             # A learning rate scheduler implements zero_grad and step. Check that this
             # is the case for the class retrieved from torch.optim.lr_scheduler
             hasattr(torch.optim.lr_scheduler.__dict__[lr_scheduler_key], 'step') and
             callable(torch.optim.lr_scheduler.__dict__[lr_scheduler_key].step))]

    if len(lr_scheduler_namespace_keys) == 0:
        raise ValueError(
            f"{lr_scheduler_str} is not a valid lr scheduler in namespaces {[_namespace.__name__ for _namespace in LR_SCHEDULER_NAMESPACES]}."
        )

    namespace, lr_scheduler_name = lr_scheduler_namespace_keys[0]
    if len(lr_scheduler_namespace_keys) > 1:
        warnings.warn(
            f"There are multiple potential matches for lr scheduler {lr_scheduler_str} ({[_lr_scheduler_name for _, _lr_scheduler_name in lr_scheduler_namespace_keys]}). "
            f"Defaulting to {lr_scheduler_name} from {namespace.__name__}.")

    lr_scheduler_class = getattr(namespace, lr_scheduler_name)
    return lr_scheduler_class


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

    # The attributes in _DICT_ATTRIBUTES are parsed to dictionaries.
    _DICT_ATTRIBUTES = ["lr_scheduler_kwargs"]

    def __post_init__(self) -> None:
        # Parse in args that could be `dict` sent in from the CLI as a string
        parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)
        # Parse string to learning rate scheduler class if needed
        self.lr_scheduler_cls = (
            _parse_lr_scheduler_class(self.lr_scheduler_cls) if isinstance(
                self.lr_scheduler_cls, str) else self.lr_scheduler_cls)


@dataclass
class OptimizerArgs:
    target_params: Union[str, TargetParamFn] = field(
        metadata={
            "help": ("Targets to be optimized."),
            "choices": TARGET_PARAM_FN_REGISTRY.get_registered_keys(),})
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
        # Parse args that could be `dict` sent in from the CLI as a string
        parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)
        # Parse optimizer name to class
        self.optimizer_cls = (
            _parse_optimizer_class(self.optimizer_cls)
            if isinstance(self.optimizer_cls, str) else self.optimizer_cls)
        # Initialize the target parametrizations
        self.target_params = (
            TARGET_PARAM_FN_REGISTRY.get(self.target_params)
            if isinstance(self.target_params, str) else self.target_params)
        if self.lr < 0:
            raise ValueError(f"Expected a positive learning rate but {self.lr} was passed.")


@dataclass
class TrainingArgs:
    optimizers_args: List[OptimizerArgs] = field(
        metadata={"help": ("Hyperparameters of the optimizers to use during training.")})
    batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for training."})
    iters: int = field(default=200, metadata={"help": "Number of training iterations."})
    loss_cls: Union[str, Type[BlockLoss]] = field(
        default="mse",
        metadata={
            "help": "Class of the loss to be used for rounding optimization.",
            "choices": BLOCK_LOSS_REGISTRY.get_registered_keys()})
    loss_kwargs: Optional[Union[Dict, str]] = field(
        default=None,
        metadata={"help": "Extra keyword arguments for the learned round loss."},
    )
    loss_scaling_factor: float = field(
        default=1.,
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

    _DICT_ATTRIBUTES = ["loss_kwargs"]

    def __post_init__(self) -> None:
        # Parse in args that could be `dict` sent in from the CLI as a string
        parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)

        for optimizer_args in self.optimizers_args:
            # Check if the optimizer has an attached learning rate scheduler
            if optimizer_args.lr_scheduler_args is not None:
                optimizer_args.lr_scheduler_args.lr_scheduler_kwargs["total_iters"] = self.iters
        # Parse amp_dtype
        self.amp_dtype = getattr(torch, self.amp_dtype) if isinstance(
            self.amp_dtype, str) else self.amp_dtype
        # Retrieve loss
        self.loss_cls = (
            BLOCK_LOSS_REGISTRY.get(self.loss_cls)
            if isinstance(self.loss_cls, str) else self.loss_cls)


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
    fast_update: bool = field(
        default=True, metadata={"help": ("Whether to use fast update with learned round.")})

    _DICT_ATTRIBUTES = ["learned_round_kwargs"]

    def __post_init__(self) -> None:
        # Parse in args that could be `dict` sent in from the CLI as a string
        parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)

        self.learned_round_param = LearnedRoundImplType(
            self.learned_round_param.upper()) if isinstance(
                self.learned_round_param, str) else self.learned_round_param


@dataclass
class Config:
    learned_round_args: LearnedRoundArgs = field(
        metadata={"help": "Learned round parametrization."})
    training_args: TrainingArgs = field(metadata={"help": "Hyperparameters for optimization."})
