# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import Callable, Dict, Generator, List, Optional, Tuple, Type

import torch
from torch import nn
import torch.nn.functional as F

from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL


class LearnedRoundLoss(ABC):

    @abstractmethod
    def __init__(self, block: nn.Module, learned_round_modules: List[nn.Module], **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        pass

    @abstractmethod
    def format_loss_components(self, *args) -> str:
        pass


def learned_round_value_init_non_linear(
    layer: nn.Module,
    learned_round_zeta: float = 1.1,
    learned_round_gamma: float = -0.1,
    **learned_round_impl_kwargs,
) -> torch.Tensor:
    floor_weight = torch.floor(layer.weight.data / layer.quant_weight().scale)
    delta = (layer.weight.data / layer.quant_weight().scale) - floor_weight
    value = -torch.log((learned_round_zeta - learned_round_gamma) /
                       (delta - learned_round_gamma) - 1)
    return value


def learned_round_value_init_linear(
    layer: nn.Module,
    **learned_round_impl_kwargs,
) -> torch.Tensor:
    value = torch.zeros_like(layer.weight.data)
    return value


LEARNED_ROUND_VALUE_INIT_MAP = {
    LearnedRoundImplType.HARD_SIGMOID.value: learned_round_value_init_non_linear,
    LearnedRoundImplType.SIGMOID.value: learned_round_value_init_non_linear,
    LearnedRoundImplType.IDENTITY.value: learned_round_value_init_linear,}


class LearnedRound(ABC):

    def __init__(
        self,
        learned_round_impl_type: LearnedRoundImplType = LearnedRoundImplType.HARD_SIGMOID,
        learned_round_value_init_fn: Optional[Callable] = None,
        **learned_round_impl_kwargs,
    ) -> None:
        self.learned_round_impl_type = learned_round_impl_type
        self.learned_round_value_init_fn = learned_round_value_init_fn
        self.learned_round_impl_kwargs = learned_round_impl_kwargs

    def learned_round_value_init(
        self,
        layer: nn.Module,
    ) -> torch.Tensor:
        # A custom initialization function for the learned round parameter can be passed
        if self.learned_round_value_init_fn is not None:
            return self.learned_round_value_init_fn(layer, **self.learned_round_impl_kwargs)
        # If not provided, the default function, as defined in LEARNED_ROUND_VALUE_INIT_MAP
        # is leveraged
        return LEARNED_ROUND_VALUE_INIT_MAP[self.learned_round_impl_type.value](
            layer, **self.learned_round_impl_kwargs)

    def _insert_learned_round_quantizer_to_layer(self, layer: nn.Module) -> None:
        value = self.learned_round_value_init(layer)
        layer.weight_quant.quant_injector = layer.weight_quant.quant_injector.let(
            float_to_int_impl_type=FloatToIntImplType.LEARNED_ROUND,
            learned_round_impl_type=self.learned_round_impl_type,
            learned_round_init=value,
            **self.learned_round_impl_kwargs,
        )
        layer.weight_quant.init_tensor_quant(preserve_state_dict=True)

    def insert_learned_round_quantizers(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, QuantWBIOL) and len(
                    self.return_learned_round_quantizers(module)) == 0:
                self._insert_learned_round_quantizer_to_layer(module)

    def return_learned_round_quantizers(self, block: nn.Module) -> List[nn.Module]:
        return [module for module in block.modules() if isinstance(module, LearnedRoundSte)]


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


class RegularisedMSELoss(LearnedRoundLoss):

    def __init__(
            self,
            module: nn.Module,
            learned_round_modules: List[nn.Module],
            weight: float = 0.01,
            max_count: int = 1000,
            b_range: Tuple = (20, 2),
            warmup: float = 0.2,
            decay_start: float = 0.0,
            **kwargs) -> None:
        # This loss operates in a layer-wise manner, so integrity needs to be checked
        assert isinstance(module, QuantWBIOL), "Regularised MSE loss can only accept a single QuantWBIOL layer."
        assert len(learned_round_modules) == 1, "Regularised MSE loss can only accept a single learned round module."

        self.weight = weight
        self.module = module
        self.loss_start = max_count * warmup
        self.temp_decay = LinearTempDecay(
            max_count,
            start_b=b_range[0],
            end_b=b_range[1],
            rel_start_decay=warmup + (1.0 - warmup) * decay_start)
        self.iter = 0
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


class MSELoss(LearnedRoundLoss):

    def __init__(self, block: nn.Module, learned_round_modules: List[nn.Module], **kwargs) -> None:
        pass

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        loss = F.mse_loss(pred, tgt)
        return loss, (loss.detach().cpu().item(),)

    def format_loss_components(self, loss: float) -> str:
        return "Loss = {:.4f}".format(loss)
