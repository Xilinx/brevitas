# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL


# TODO: Rename to block loss
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


def return_learned_round_quantizers(block: nn.Module) -> List[nn.Module]:
    return [module for module in block.modules() if isinstance(module, LearnedRoundSte)]


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


class MSELoss(BlockLoss):

    def __init__(self, block: nn.Module, **kwargs) -> None:
        pass

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        loss = F.mse_loss(pred, tgt)
        return loss, (loss.detach().cpu().item(),)

    def format_loss_components(self, loss: float) -> str:
        return "Loss = {:.4f}".format(loss)
