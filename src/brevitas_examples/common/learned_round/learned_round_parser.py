# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import re
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
import warnings

from accelerate.utils.operations import send_to_device
from datasets import Dataset
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from brevitas import config
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.optim.sign_sgd import SignSGD
from brevitas.quant_tensor import QuantTensor
from brevitas_examples.common.learned_round.learned_round_method import LearnedRound
from brevitas_examples.common.learned_round.learned_round_method import LearnedRoundLoss
from brevitas_examples.common.learned_round.learned_round_method import MSELoss
from brevitas_examples.common.learned_round.learned_round_method import RegularisedMSELoss
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundOptimizer

LEARNED_ROUND_MAP = {
    "linear_round": LearnedRoundImplType.IDENTITY,
    "hard_sigmoid_round": LearnedRoundImplType.HARD_SIGMOID,
    "sigmoid_round": LearnedRoundImplType.SIGMOID,}
LEARNED_ROUND_LOSS_MAP = {
    "mse": MSELoss,
    "regularised_mse": RegularisedMSELoss,}
OPTIMIZER_MAP = {
    "sign_sgd": SignSGD,}
LR_SCHEDULER_MAP = {}


def parse_learned_round(learned_round_str: str) -> LearnedRound:
    if learned_round_str not in LEARNED_ROUND_MAP:
        raise ValueError(f"Learned round method {learned_round_str} is not available.")
    return LearnedRound(learned_round_impl_type=LEARNED_ROUND_MAP[learned_round_str])


def parse_learned_round_loss_class(learned_round_loss_str: str) -> Type[LearnedRoundLoss]:
    if learned_round_loss_str not in LEARNED_ROUND_LOSS_MAP:
        raise ValueError(f"Learned round loss {learned_round_loss_str} is not available.")
    return LEARNED_ROUND_LOSS_MAP[learned_round_loss_str]


def parse_optimizer_class(optimizer_str: str) -> Type[Optimizer]:
    if optimizer_str in OPTIMIZER_MAP:
        optimizer_class = OPTIMIZER_MAP[optimizer_str]
    else:
        optimizer_keys = [
            optimizer_key for optimizer_key in torch.optim.__dict__.keys() if (
                optimizer_key.lower().startswith(optimizer_str.lower()) and
                torch.optim.__dict__[optimizer_key] != Optimizer and
                isinstance(torch.optim.__dict__[optimizer_key], type) and
                issubclass(torch.optim.__dict__[optimizer_key], Optimizer))]
        if len(optimizer_keys) == 0:
            raise ValueError(f"{optimizer_str} is not a valid optimizer.")
        else:
            if len(optimizer_keys) > 1:
                warnings.warn(
                    f"There are multiple potential matches for optimizer {optimizer_str}. "
                    f"Defaulting to {optimizer_keys[0]}")
            optimizer_class = getattr(torch.optim, optimizer_keys[0])

    return optimizer_class


def parse_lr_scheduler_class(lr_scheduler_str: str) -> Type[LRScheduler]:
    if lr_scheduler_str in LR_SCHEDULER_MAP:
        lr_scheduler_class = LR_SCHEDULER_MAP[lr_scheduler_str]
    else:
        lr_scheduler_keys = [
            lr_scheduler_key for lr_scheduler_key in torch.optim.lr_scheduler.__dict__.keys() if (
                lr_scheduler_key.lower().startswith(lr_scheduler_str.lower()) and
                torch.optim.lr_scheduler.__dict__[lr_scheduler_key] != LRScheduler and
                isinstance(torch.optim.lr_scheduler.__dict__[lr_scheduler_key], type) and
                issubclass(torch.optim.lr_scheduler.__dict__[lr_scheduler_key], LRScheduler))]
        print(lr_scheduler_keys)
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
