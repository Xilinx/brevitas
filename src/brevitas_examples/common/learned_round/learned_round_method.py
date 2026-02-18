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


class TrainingMethod(Protocol[T_config]):
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
    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        pass

    @abstractmethod
    def format_loss_components(self, *args) -> str:
        pass


TRAINING_METHODS_REGISTRY = Registry[Type[TrainingMethod]]('TrainingMethod Registry')

# Registries for implementations of learned round components
BLOCK_LOSS_REGISTRY = Registry[Type[BlockLoss]]('BlockLoss Registry')
TARGET_PARAM_FN_REGISTRY = Registry[TargetParamFn]('TargetParamFn Registry')
LEARNED_ROUND_INIT_FN_REGISTRY = Registry[LearnedRoundInitFn]('LearnedRoundInitFn Registry')


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


@BLOCK_LOSS_REGISTRY.register(names="regularised_mse")
class RegularisedMSELoss(BlockLoss):

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
        assert isinstance(module, QuantWBIOL), "Regularised MSE loss can only accept a single QuantWBIOL layer."
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
        assert len(learned_round_modules) == 1, "Regularised MSE loss can only accept a single learned round module."
        self.learned_round_module = learned_round_modules[0]

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        self.iter += 1

        rec_loss = F.mse_loss(pred, tgt, reduction='none').sum(1).mean()

        if self.iter < self.loss_start:
            b = self.temp_decay(self.iter)
            round_loss = 0.
        else:  # 1 - |(h-0.5)*2|**b
            b = self.temp_decay(self.iter)
            round_vals = self.learned_round_module.learned_round_impl(
                self.learned_round_module.value)
            round_loss = self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        return total_loss, (total_loss, rec_loss, round_loss, b)

    def format_loss_components(self, loss: float, rec_loss: float, round_loss: float, b) -> str:
        return "Loss = {:.4f}, rec_loss = {:.4f}, round_loss = {:.4f}, b = {:.4f}".format(
            loss,
            rec_loss.detach().cpu().item(),
            round_loss if isinstance(round_loss, float) else round_loss.detach().cpu().item(),
            b)


@BLOCK_LOSS_REGISTRY.register(names="mse")
class MSELoss(BlockLoss):

    def __init__(self, block: nn.Module, **kwargs) -> None:
        pass

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        loss = F.mse_loss(pred, tgt)
        return loss, (loss.detach().cpu().item(),)

    def format_loss_components(self, loss: float) -> str:
        return "Loss = {:.4f}".format(loss)


# Both `get_round_parameters` and `get_scale_parameters` are meant to be passed as the argument `get_target`
# of `_get_target_parameters`, which iterates over the modules of a model in a recursive function.
# The return value indicates whether the submodules of a given module need to be skipped.
# For instance, for `get_round_parameters`, when a LearnedRoundSte module is found, the
# learned round parameters are added to `state_dict` and `True` is returned to stop the recursion.
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
    fast_update: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use fast update with learned round. `fast_update=True` requires implementing additional methods in the custom `Cache`."
            )})

    _DICT_ATTRIBUTES = ["learned_round_kwargs"]

    def __post_init__(self) -> None:
        # Parse in args that could be `dict` sent in from the CLI as a string
        parse_dataclass_dicts(self, self._DICT_ATTRIBUTES)

        self.learned_round_param = LearnedRoundImplType(
            self.learned_round_param.upper()) if isinstance(
                self.learned_round_param, str) else self.learned_round_param


@TRAINING_METHODS_REGISTRY.register(names="learned_round")
class LearnedRoundTrainer(TrainingMethod[LearnedRoundArgs]):

    def __init__(self, config: LearnedRoundArgs) -> None:
        self.config = config

    def _insert_learned_round_quantizers(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, QuantWBIOL) and len([
                    m for m in module.modules() if isinstance(m, LearnedRoundSte)]) == 0:
                learned_round_init_fn = LEARNED_ROUND_INIT_FN_REGISTRY.get(
                    self.config.learned_round_param.value)
                value = learned_round_init_fn(module, **self.config.learned_round_kwargs)
                module.weight_quant.quant_injector = module.weight_quant.quant_injector.let(
                    float_to_int_impl_type=FloatToIntImplType.LEARNED_ROUND,
                    learned_round_impl_type=self.config.learned_round_param,
                    learned_round_init=value,
                    **self.config.learned_round_kwargs,
                )
                module.weight_quant.init_tensor_quant(preserve_state_dict=True)

    def prepare_model(self, model: nn.Module) -> None:
        # Insert learned round quantizers within the appropiate model blocks
        self._insert_learned_round_quantizers(model)
