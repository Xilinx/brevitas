# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import torch
from torch.optim.optimizer import Optimizer

from brevitas.utils.python_utils import parse_dataclass_dicts
from brevitas_examples.common.learned_round.learned_round_method import BLOCK_LOSS_REGISTRY
from brevitas_examples.common.learned_round.learned_round_method import BlockLoss
from brevitas_examples.common.learned_round.learned_round_method import LearnedRoundArgs
from brevitas_examples.common.learned_round.learned_round_method import TARGET_PARAM_FN_REGISTRY
from brevitas_examples.common.learned_round.learned_round_method import TargetParamFn
from brevitas_examples.common.trainer_utils import parse_lr_scheduler_class
from brevitas_examples.common.trainer_utils import parse_optimizer_class


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
            parse_lr_scheduler_class(self.lr_scheduler_cls) if isinstance(
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
            parse_optimizer_class(self.optimizer_cls)
            if isinstance(self.optimizer_cls, str) else self.optimizer_cls)
        # Initialize the target parametrizations
        self.target_params = (
            TARGET_PARAM_FN_REGISTRY.get(self.target_params)
            if isinstance(self.target_params, str) else self.target_params)
        if self.lr < 0:
            raise ValueError(f"Expected a positive learning rate but {self.lr} was passed.")


@dataclass
class LossArgs:
    cls: Union[str, Type[BlockLoss]] = field(
        default="mse",
        metadata={
            "help": "Class of the loss to be used for blockwise optimization.",
            "choices": BLOCK_LOSS_REGISTRY.get_registered_keys()})
    kwargs: Optional[Union[Dict, str]] = field(
        default=None,
        metadata={"help": "Extra keyword arguments for the loss."},
    )

    _DICT_ATTRIBUTES = ["kwargs"]

    def __post_init__(self) -> None:
        # Parse in args that could be `dict` sent in from the CLI as a string
        parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)
        # Retrieve loss
        self.cls = (BLOCK_LOSS_REGISTRY.get(self.cls) if isinstance(self.cls, str) else self.cls)


@dataclass
class TrainingArgs:
    optimizers_args: List[OptimizerArgs] = field(
        metadata={"help": ("Hyperparameters of the optimizers to use during training.")})
    batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for training."})
    iters: int = field(default=200, metadata={"help": "Number of training iterations."})
    losses_args: List[LossArgs] = field(
        default_factory=list, metadata={"help": "Losses to use during blockwise training."})
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
    fast_update: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use fast update with block optimization. `fast_update=True` requires implementing additional methods in the custom `Cache`."
            )})

    def __post_init__(self) -> None:
        # Verify that at least one loss function was provided
        if len(self.losses_args) == 0:
            raise ValueError("At least one loss function needs to be provided for training.")
        # Verify that at least one optimizer was provided
        if len(self.optimizers_args) == 0:
            raise ValueError("At least one optimizer needs to be provided for training.")
        for optimizer_args in self.optimizers_args:
            # Check if the optimizer has an attached learning rate scheduler
            if optimizer_args.lr_scheduler_args is not None:
                optimizer_args.lr_scheduler_args.lr_scheduler_kwargs["total_iters"] = self.iters
        # Parse amp_dtype
        self.amp_dtype = getattr(torch, self.amp_dtype) if isinstance(
            self.amp_dtype, str) else self.amp_dtype


T_config = TypeVar("T_config")


@dataclass
class HandlerSpec(Generic[T_config]):
    name: str
    config: T_config


@dataclass
class TrainerConfig:
    training_args: TrainingArgs = field(metadata={"help": "Hyperparameters for optimization."})
    training_handlers: List[HandlerSpec] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of training handlers to be applied during training. Each handler's `name` needs to be registered in `TRAINING_METHODS_REGISTRY`."
            )})
