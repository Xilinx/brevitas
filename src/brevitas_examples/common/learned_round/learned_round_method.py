# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from typing import Generator, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from brevitas.core.function_wrapper.learned_round import AutoRoundSte
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL


class StopFwdException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""
    pass


class LearnedRoundLoss(ABC):

    @abstractmethod
    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        pass

    @abstractmethod
    def format_loss_components(self, *args) -> str:
        pass


class LearnedRound(ABC):

    def __init__(self, iters: int = 200, **kwargs) -> None:
        self.iters = iters

    def _insert_learned_round_quantizer(self, block: nn.Module) -> None:
        for module in block.modules():
            if isinstance(module, QuantWBIOL) and len(
                    self._find_learned_round_modules(module)) == 0:
                self._insert_learned_round_quantizer_to_layer(module)
                module.weight_quant.init_tensor_quant(preserve_state_dict=True)

    @abstractmethod
    def _insert_learned_round_quantizer_to_layer(self, layer: nn.Module) -> None:
        pass

    @abstractmethod
    def _is_learned_round_module(self, module: nn.Module) -> bool:
        pass

    @abstractmethod
    def _instantiate_loss(
            self, block: nn.Module, learned_round_modules: List[nn.Module]) -> LearnedRoundLoss:
        pass

    def _find_learned_round_modules(self, block: nn.Module) -> List[nn.Module]:
        round_modules = []
        for module in block.modules():
            if self._is_learned_round_module(module):
                round_modules.append(module)
        return round_modules

    def learned_round_iterator(
            self,
            blocks: List[nn.Module]) -> Generator[nn.Module, LearnedRoundLoss, List[nn.Module]]:
        for block in blocks:
            # Insert learned round quantizers into the appropiate submodules
            self._insert_learned_round_quantizer(block)
            # Freeze block parameters
            for params in block.parameters():
                params.requires_grad = False
            # Retrieve learned round modules
            learned_round_modules = self._find_learned_round_modules(block)
            # Enable gradient tracking in learned round modules
            for round_module in learned_round_modules:
                for params in round_module.parameters():
                    params.requires_grad = True
            block_loss = self._instantiate_loss(block, learned_round_modules)
            yield block, block_loss, learned_round_modules


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


class AdaRoundLoss(LearnedRoundLoss):

    def __init__(
            self,
            module: nn.Module,
            learned_round_modules: List[nn.Module],
            weight: float = 0.01,
            max_count: int = 1000,
            b_range: Tuple = (20, 2),
            warmup: float = 0.2,
            decay_start: float = 0.0) -> None:
        super().__init__()
        # AdaRound operates in a layer-wise manner, so integrity needs to be checked
        assert isinstance(module, QuantWBIOL), "AdaRound can only accept a single QuantWBIOL layer."
        assert len(learned_round_modules) == 1, "AdaRound can only accept a single learned round module."

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
            round_loss = 0
        else:  # 1 - |(h-0.5)*2|**b
            b = self.temp_decay(self.iter)
            round_vals = self.learned_round_module.p_forward()
            round_loss = self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        return total_loss, (total_loss, rec_loss, round_loss, b)

    def format_loss_components(self, loss: float, rec_loss: float, round_loss: float, b) -> str:
        return "loss = {:.4f}, rec_loss = {:.4f}, round_loss = {:.4f}, b = {:.4f}".format(
            loss, rec_loss, round_loss, b)


class AdaRound(LearnedRound):

    def __init__(
            self,
            iters: int = 200,
            weight: float = 0.01,
            b_range: Tuple = (20, 2),
            warmup: float = 0.2,
            decay_start: float = 0.0,
            **kwargs,
    ) -> None:
        super().__init__(iters, **kwargs)
        # Loss-related configuration
        self.weight = weight
        self.b_range = b_range
        self.warmup = warmup
        self.decay_start = decay_start

    def _is_learned_round_module(self, module: nn.Module) -> bool:
        return isinstance(module, LearnedRoundSte)

    def _insert_learned_round_quantizer_to_layer(
            self,
            layer: nn.Module,
            learned_round_zeta: float = 1.1,
            learned_round_gamma: float = -0.1) -> None:
        floor_weight = torch.floor(layer.weight.data / layer.quant_weight().scale)
        delta = (layer.weight.data / layer.quant_weight().scale) - floor_weight
        value = -torch.log((learned_round_zeta - learned_round_gamma) /
                           (delta - learned_round_gamma) - 1)
        layer.weight_quant.quant_injector = layer.weight_quant.quant_injector.let(
            float_to_int_impl_type=FloatToIntImplType.LEARNED_ROUND,
            learned_round_impl_type=LearnedRoundImplType.HARD_SIGMOID,
            learned_round_gamma=learned_round_gamma,
            learned_round_zeta=learned_round_zeta,
            learned_round_init=value)

    def _instantiate_loss(
            self, block: nn.Module, learned_round_modules: List[nn.Module]) -> AdaRoundLoss:
        return AdaRoundLoss(
            block,
            learned_round_modules,
            max_count=self.iters,
            weight=self.weight,
            warmup=self.warmup,
            decay_start=self.decay_start,
        )


class AutoRoundLoss(LearnedRoundLoss):

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        loss = F.mse_loss(pred, tgt)
        return loss, (loss,)

    def format_loss_components(self, loss: float) -> str:
        return "loss = {:.4f}".format(loss)


class AutoRound(LearnedRound):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _is_learned_round_module(self, module: nn.Module) -> bool:
        return isinstance(module, AutoRoundSte)

    def _insert_learned_round_quantizer_to_layer(self, layer: nn.Module) -> None:
        value = torch.zeros_like(layer.weight.data)
        layer.weight_quant.quant_injector = layer.weight_quant.quant_injector.let(
            float_to_int_impl_type=FloatToIntImplType.AUTO_ROUND,
            learned_round_init=value,
        )

    def _instantiate_loss(
            self, block: nn.Module, learned_round_modules: List[nn.Module]) -> AutoRoundLoss:
        return AutoRoundLoss()
