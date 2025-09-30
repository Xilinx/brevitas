# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from dataclasses import field
from enum import auto
import json
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import OrderedDict
from typing import Type
from typing import Union
import warnings

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.optim.sign_sgd import SignSGD
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
from brevitas.utils.python_utils import AutoName
from brevitas_examples.common.learned_round.learned_round_method import BlockLoss
from brevitas_examples.common.learned_round.learned_round_method import MSELoss
from brevitas_examples.common.learned_round.learned_round_method import RegularisedMSELoss

OPTIMIZER_MAP = {
    "sign_sgd": SignSGD,}
LR_SCHEDULER_MAP = {}


# TODO (pml): Is it possible to remove this boilerplate?
class TargetParametrizations(AutoName):
    SCALES = auto()
    LEARNED_ROUND = auto()

    @property
    def param_fn(self) -> Callable[[nn.Module, OrderedDict, str], bool]:
        return {
            TargetParametrizations.SCALES.value: get_scale_parameters,
            TargetParametrizations.LEARNED_ROUND.value: get_round_parameters}[self.value]


class BlockLossType(AutoName):
    MSE = auto()
    REGULARISED_MSE = auto()

    @property
    def loss_class(self) -> Type[BlockLoss]:
        return {
            BlockLossType.MSE.value: MSELoss,
            BlockLossType.REGULARISED_MSE.value: RegularisedMSELoss}[self.value]


# Both `get_round_parameters` and `get_scale_parameters` are meant to be passed as the argument `get_target`
# of `_get_target_parameters`, which iterates over the modules of a model in a recursive function.
# In the case of `get_round_parameters` the return value indicates whether the submodules of a given module
# need to be explored. Therefore, when a LearnedRoundSte module is found, the learned round parameters
# are added to `state_dict` and `True` is returned to stop the recursion.
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


#TODO (pml): Add license from Nanotron
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


def _parse_dataclass_dicts(data_cls, dict_attributes: List[str]) -> None:
    """
    Parses the strings in `dict_attributes` of dataclass `data_cls` to dictionaries.
    """
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


def _parse_optimizer_class(optimizer_str: str) -> Type[Optimizer]:
    if optimizer_str in OPTIMIZER_MAP:
        optimizer_class = OPTIMIZER_MAP[optimizer_str]
    else:
        optimizer_keys = [
            optimizer_key for optimizer_key in torch.optim.__dict__.keys()
            # Check for making sure that only valid optimizer implementations are
            # retrieved, when matching with the string passed by the user
            if (
                # Verify that the key stars with the one passed by the user
                optimizer_key.lower() == optimizer_str.lower() and
                # Verify that key corresponds to a class
                isinstance(torch.optim.__dict__[optimizer_key], type) and
                # Make sure the abstract class is not used
                optimizer_key != "Optimizer" and
                # An optimizer implements zero_grad and step. Check that this
                # is the case for the class retrieved from torch.optim
                hasattr(torch.optim.__dict__[optimizer_key], 'step') and
                callable(torch.optim.__dict__[optimizer_key].step) and
                hasattr(torch.optim.__dict__[optimizer_key], 'zero_grad') and
                callable(torch.optim.__dict__[optimizer_key].zero_grad))]
        if len(optimizer_keys) == 0:
            raise ValueError(f"{optimizer_str} is not a valid optimizer.")
        else:
            if len(optimizer_keys) > 1:
                warnings.warn(
                    f"There are multiple potential matches for optimizer {optimizer_str}. "
                    f"Defaulting to {optimizer_keys[0]}")
            optimizer_class = getattr(torch.optim, optimizer_keys[0])

    return optimizer_class


def _parse_lr_scheduler_class(lr_scheduler_str: str) -> Type:
    if lr_scheduler_str in LR_SCHEDULER_MAP:
        lr_scheduler_class = LR_SCHEDULER_MAP[lr_scheduler_str]
    else:
        lr_scheduler_keys = [
            lr_scheduler_key for lr_scheduler_key in torch.optim.lr_scheduler.__dict__.keys()
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
        if len(lr_scheduler_keys) == 0:
            warnings.warn(
                f"There are no matches for LR scheduler {lr_scheduler_str}. "
                f"No LR scheduler is going to be used.")
            lr_scheduler_class = None
        else:
            if len(lr_scheduler_keys) > 1:
                warnings.warn(
                    f"There are multiple potential matches for LR scheduler {lr_scheduler_str}."
                    f"Defaulting to {lr_scheduler_keys[0]}")
            lr_scheduler_class = getattr(torch.optim.lr_scheduler, lr_scheduler_keys[0])

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
        _parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)
        # Parse string to learning rate scheduler class if needed
        self.lr_scheduler_cls = (
            _parse_lr_scheduler_class(self.lr_scheduler_cls) if isinstance(
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
        # Parse args that could be `dict` sent in from the CLI as a string
        _parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)
        # Parse optimizer name to class
        self.optimizer_cls = (
            _parse_optimizer_class(self.optimizer_cls)
            if isinstance(self.optimizer_cls, str) else self.optimizer_cls)
        if self.lr < 0:
            raise ValueError(f"Expected a positive learning rate but {self.lr} was passed.")


@dataclass
class TrainingArgs:
    optimizers_args: List[OptimizerArgs] = field(
        metadata={"help": ("Hyperparameters of the optimizers to use during training.")})
    optimizers_targets: List[Union[str, TargetParametrizations]] = field(
        metadata={
            "help": ("Targets to be optimized."),
            "choices": [
                optimizer_target.value.lower() for optimizer_target in TargetParametrizations],})
    batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for training."})
    iters: int = field(default=200, metadata={"help": "Number of training iterations."})
    loss_cls: Union[str, Type[BlockLoss]] = field(
        default="mse",
        metadata={
            "help": "Class of the loss to be used for rounding optimization.",
            "choices": [block_loss_type.value.lower() for block_loss_type in BlockLossType]})
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
        _parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)

        for optimizer_args in self.optimizers_args:
            # Check if the optimizer has an attached learning rate scheduler
            if optimizer_args.lr_scheduler_args is not None:
                optimizer_args.lr_scheduler_args.lr_scheduler_kwargs["total_iters"] = self.iters
        # Initialize the target parametrizations
        self.optimizers_targets = [
            TargetParametrizations(optimizer_target.upper())
            if isinstance(optimizer_target, str) else optimizer_target
            for optimizer_target in self.optimizers_targets]
        # Parse amp_dtype
        self.amp_dtype = getattr(torch, self.amp_dtype) if isinstance(
            self.amp_dtype, str) else self.amp_dtype
        # Retrieve loss
        self.loss_cls = (
            BlockLossType(self.loss_cls.upper()).loss_class
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
        _parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)

        self.learned_round_param = LearnedRoundImplType(
            self.learned_round_param.upper()) if isinstance(
                self.learned_round_param, str) else self.learned_round_param


@dataclass
class Config:
    learned_round_args: LearnedRoundArgs = field(
        metadata={"help": "Learned round parametrization."})
    training_args: TrainingArgs = field(metadata={"help": "Hyperparameters for optimization."})
