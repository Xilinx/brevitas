from argparse import Namespace
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
import itertools
import json
from typing import Any, Callable, Dict, Generator, List, Optional, OrderedDict, Tuple, Type, Union

import torch
from torch import nn
from torch.optim.optimizer import Optimizer

from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
from brevitas_examples.common.learned_round.learned_round_method import LearnedRoundLoss
from brevitas_examples.common.learned_round.learned_round_parser import LEARNED_ROUND_MAP
from brevitas_examples.common.learned_round.learned_round_parser import \
    parse_learned_round_loss_class
from brevitas_examples.common.learned_round.learned_round_parser import parse_lr_scheduler_class
from brevitas_examples.common.learned_round.learned_round_parser import parse_optimizer_class


#TODO: Add license
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


class TargetParametrizations(Enum):
    SCALES = "scales"
    LEARNED_ROUND = "learned_round"


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


def get_target_parameters(
        block: nn.Module, get_target: Callable[[nn.Module, OrderedDict, str],
                                               bool]) -> 'OrderedDict[str, nn.Parameter]':
    state_dict = OrderedDict()

    def _get_target_parameters(module: nn.Module, prefix: str = "") -> None:
        # Base case
        if get_target(module, state_dict, prefix):
            # Early stoppping
            return
        for child_name, child_module in module.named_children():
            _get_target_parameters(
                child_module, f"{prefix}.{child_name}" if len(prefix) > 0 else f"{child_name}")

    # Run recursion from block
    _get_target_parameters(block)
    return state_dict


def get_scale_parameters(model: nn.Module) -> List[nn.Parameter]:
    scale_parameters = []

    def _get_scale_parameters(module: nn.Module):
        for module_child in module.children():
            if isinstance(module, WeightQuantProxyFromInjectorBase):
                for submodule_name, submodule in module_child.named_parameters():
                    if submodule_name.endswith('scaling_impl.value'):
                        scale_parameters.append(submodule)
            else:
                _get_scale_parameters(module_child)

    # Run recursion from root module
    _get_scale_parameters(model)
    return scale_parameters


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
    use_best_model: bool = field(
        default=True,
        metadata={
            "help":
                ("Whether to use the best setting of the learned round found "
                 "during training.")})
    use_amp: bool = field(
        default=False,
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
    learned_round_param: Union[str] = field(
        default="linear_round",
        metadata={
            "help": "Defines the functional form of the learned round parametrization.",
            "choices": list(LEARNED_ROUND_MAP.keys())})
    learned_round_kwargs: Optional[Union[Dict, str]] = field(
        default=None,
        metadata={"help": "Extra keyword arguments for the learned round parametrization."},
    )
    loss_cls: Union[str, Type[LearnedRoundLoss]] = field(
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

        self.loss_cls = (
            parse_learned_round_loss_class(self.loss_cls)
            if isinstance(self.loss_cls, str) else self.loss_cls)


@dataclass
class Config:
    learned_round_args: LearnedRoundArgs = field(
        metadata={"help": "Learned round parametrization."})
    training_args: TrainingArgs = field(metadata={"help": "Hyperparameters for optimization."})


# TODO: Move to a file of helper methods
def build_optimizer_and_scheduler(
    params: List[nn.Parameter],
    optimizer_args: OptimizerArgs,
) -> Tuple[Optimizer, Optional[Any]]:
    # Instantiate optimizer
    optimizer = optimizer_args.optimizer_cls(
        params=params, lr=optimizer_args.lr, **optimizer_args.optimizer_kwargs)
    # Instantiate learning rate schedu
    lr_scheduler_args = optimizer_args.lr_scheduler_args
    lr_scheduler = (
        lr_scheduler_args.lr_scheduler_cls(optimizer, **lr_scheduler_args.lr_scheduler_kwargs)
        if lr_scheduler_args is None else None)
    return optimizer, lr_scheduler


def parse_args_to_dataclass(args: Namespace) -> LearnedRoundArgs:
    config_dict = {
        "learned_round_args": {
            "learned_round_param": args.learned_round,
            "learned_round_kwargs": None,
            "loss_cls": "mse",
            "loss_kwargs": None,
            "fast_update": args.learned_round_fast_update,},
        "training_args": {
            "optimizers_args": [{
                "optimizer_cls": "sign_sgd",
                "lr": args.learned_round_lr,
                "optimizer_kwargs": {},
                "lr_scheduler_args": {
                    "lr_scheduler_cls": "linear",
                    "lr_scheduler_kwargs": '{"start_factor": 1.0, "end_factor": 0.0}'}},
                                {
                                    "optimizer_cls": "sgd",
                                    "lr": args.learned_round_scale_lr,
                                    "optimizer_kwargs": {
                                        "momentum": args.learned_round_scale_momentum,},
                                    "lr_scheduler_args": None,}],
            "block_name_attribute":
                args.gpxq_block_name,
            "optimizers_targets": ["learned_round"] +
                                  (["scales"] if args.learned_round_scale else []),
            "batch_size":
                8,
            "iters":
                200,
            "use_best_model":
                True,
            "use_amp":
                False,
            "amp_dtype":
                "float16",}}
    from dacite import from_dict
    config = from_dict(data_class=Config, data=config_dict)
    return config
