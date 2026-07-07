# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Dict
from typing import List
from typing import Optional
from typing import OrderedDict
from typing import Protocol
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F

from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
from brevitas.utils.python_utils import parse_dataclass_dicts
from brevitas.utils.python_utils import Registry

T_config = TypeVar("T_config")


class TrainingHandler(Protocol[T_config]):
    """Optional extension that can modify model for a specific optimization method."""

    def __init__(self, config: T_config) -> None:
        ...

    def prepare_model(
        self,
        model: torch.nn.Module,
    ) -> None:
        ...


class TargetParamFn(Protocol):

    def __call__(self, module: nn.Module, state_dict: OrderedDict, prefix: str = "") -> bool:
        ...


class LearnedRoundInitFn(Protocol):

    def __call__(self, module: nn.Module, **kwargs) -> torch.Tensor:
        ...


class BlockLoss(ABC):

    @abstractmethod
    def __init__(self, block: nn.Module, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pass


TRAINING_HANDLERS_REGISTRY = Registry[Type[TrainingHandler]]('TrainingHandlers Registry')

# Registries for implementations of learned round components
BLOCK_LOSS_REGISTRY = Registry[Type[BlockLoss]]('BlockLoss Registry')
TARGET_PARAM_FN_REGISTRY = Registry[TargetParamFn]('TargetParamFn Registry')
LEARNED_ROUND_INIT_FN_REGISTRY = Registry[LearnedRoundInitFn]('LearnedRoundInitFn Registry')


def insert_learned_round_quantizers(
        model: nn.Module, learned_round_param: LearnedRoundImplType, **kwargs) -> None:
    for module in model.modules():
        if isinstance(module, QuantWBIOL) and len([
                m for m in module.modules() if isinstance(m, LearnedRoundSte)]) == 0:
            learned_round_init_fn = LEARNED_ROUND_INIT_FN_REGISTRY.get(learned_round_param.value)
            value = learned_round_init_fn(module, **kwargs)
            module.weight_quant.quant_injector = module.weight_quant.quant_injector.let(
                float_to_int_impl_type=FloatToIntImplType.LEARNED_ROUND,
                learned_round_impl_type=learned_round_param,
                learned_round_init=value,
                **kwargs,
            )
            module.weight_quant.init_tensor_quant(preserve_state_dict=True)


def return_learned_round_quantizers(block: nn.Module) -> List[nn.Module]:
    return [module for module in block.modules() if isinstance(module, LearnedRoundSte)]


@LEARNED_ROUND_INIT_FN_REGISTRY.register(
    names=[LearnedRoundImplType.HARD_SIGMOID.value, LearnedRoundImplType.SIGMOID.value])
def learned_round_value_init_non_linear(
    layer: nn.Module,
    learned_round_zeta: float = 1.1,
    learned_round_gamma: float = -0.1,
    **kwargs,
) -> torch.Tensor:
    floor_weight = torch.floor(layer.weight.data / layer.quant_weight().scale)
    delta = (layer.weight.data / layer.quant_weight().scale) - floor_weight
    value = -torch.log((learned_round_zeta - learned_round_gamma) /
                       (delta - learned_round_gamma) - 1)
    return value


@LEARNED_ROUND_INIT_FN_REGISTRY.register(names=LearnedRoundImplType.IDENTITY.value)
def learned_round_value_init_linear(
    layer: nn.Module,
    **kwargs,
) -> torch.Tensor:
    value = torch.zeros_like(layer.weight.data)
    return value


class LinearTempDecay:

    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


@BLOCK_LOSS_REGISTRY.register(names="round_reg")
class RoundRegularisationLoss(BlockLoss):

    def __init__(
            self,
            module: nn.Module,
            weight: float = 0.01,
            max_count: int = 1000,
            b_range: Tuple = (20, 2),
            warmup: float = 0.2,
            decay_start: float = 0.0,
            **kwargs) -> None:
        # This loss operates in a layer-wise manner, so integrity needs to be checked
        assert isinstance(module, QuantWBIOL), "Regularised round loss can only accept a single QuantWBIOL layer."
        self.weight = weight
        self.module = module
        self.loss_start = max_count * warmup
        self.temp_decay = LinearTempDecay(
            max_count,
            start_b=b_range[0],
            end_b=b_range[1],
            rel_start_decay=warmup + (1.0 - warmup) * decay_start)
        self.iter = 0
        # Retrieve learned round module for block
        learned_round_modules = return_learned_round_quantizers(module)
        assert len(learned_round_modules) == 1, "Regularised round loss can only accept a single learned round module."
        self.learned_round_module = learned_round_modules[0]

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        self.iter += 1

        if self.iter < self.loss_start:
            b = self.temp_decay(self.iter)
            round_loss = torch.tensor(0., device=pred.device, dtype=pred.dtype)
        else:  # 1 - |(h-0.5)*2|**b
            b = self.temp_decay(self.iter)
            round_vals = self.learned_round_module.learned_round_impl(
                self.learned_round_module.value)
            round_loss = self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()

        return round_loss


@BLOCK_LOSS_REGISTRY.register(names="mse")
class MSELoss(BlockLoss):

    def __init__(
            self,
            block: nn.Module,
            reduction: Optional[str] = None,
            dim: Optional[int] = None,
            **kwargs) -> None:
        self.reduction = reduction
        self.dim = dim

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        if self.reduction is None:
            return F.mse_loss(pred, tgt)
        return F.mse_loss(pred, tgt, reduction=self.reduction).sum(self.dim).mean()


# Both `get_round_parameters` and `get_scale_parameters` are meant to be passed as the argument `get_target`
# of `_get_target_parameters`, which iterates over the modules of a model in a recursive function.
# The return value indicates whether the submodules of a given module need to be skipped.
@TARGET_PARAM_FN_REGISTRY.register(names="learned_round")
def get_round_parameters(module: nn.Module, state_dict: OrderedDict, prefix: str = "") -> bool:
    if isinstance(module, LearnedRoundSte):
        for param_name, param in module.named_parameters():
            state_dict[f"{prefix}.{param_name}"] = param
        # Early stoppping
        return True
    return False


@TARGET_PARAM_FN_REGISTRY.register(names="scales")
def get_scale_parameters(module: nn.Module, state_dict: OrderedDict, prefix: str = "") -> bool:
    if isinstance(module, WeightQuantProxyFromInjectorBase):
        for param_name, param in module.named_parameters():
            if param_name.endswith('scaling_impl.value'):
                state_dict[f"{prefix}.{param_name}"] = param
        # Early stoppping
        return True
    return False


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

    _DICT_ATTRIBUTES = ["learned_round_kwargs"]

    def __post_init__(self) -> None:
        # Parse in args that could be `dict` sent in from the CLI as a string
        parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)

        self.learned_round_param = LearnedRoundImplType(
            self.learned_round_param.upper()) if isinstance(
                self.learned_round_param, str) else self.learned_round_param


@TRAINING_HANDLERS_REGISTRY.register(names="learned_round")
class LearnedRoundTrainer(TrainingHandler[LearnedRoundArgs]):

    def __init__(self, config: LearnedRoundArgs) -> None:
        self.config = config

    def prepare_model(self, model: nn.Module) -> None:
        # Insert learned round quantizers within the appropiate model blocks
        insert_learned_round_quantizers(
            model=model,
            learned_round_param=self.config.learned_round_param,
            **self.config.learned_round_kwargs,
        )
