# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Type

import torch
from torch.optim.optimizer import Optimizer

from brevitas import optim

OPTIMIZER_NAMESPACES = [torch.optim, optim]
LR_SCHEDULER_NAMESPACES = [torch.optim.lr_scheduler]


def parse_optimizer_class(optimizer_str: str) -> Type[Optimizer]:
    optimizer_class = None
    for namespace in OPTIMIZER_NAMESPACES:
        if (optimizer_class := getattr(namespace, optimizer_str, None)) is not None:
            # Stop on first match
            break

    if optimizer_class is None:
        raise ValueError(
            f"{optimizer_str} is not a valid optimizer in namespaces {[_namespace.__name__ for _namespace in OPTIMIZER_NAMESPACES]}."
        )
    return optimizer_class


def parse_lr_scheduler_class(lr_scheduler_str: str) -> Type:
    lr_scheduler_class = None
    for namespace in LR_SCHEDULER_NAMESPACES:
        if (lr_scheduler_class := getattr(namespace, lr_scheduler_str, None)) is not None:
            # Stop on first match
            break

    if lr_scheduler_class is None:
        raise ValueError(
            f"{lr_scheduler_str} is not a valid learning rate scheduler in namespaces {[_namespace.__name__ for _namespace in LR_SCHEDULER_NAMESPACES]}."
        )
    return lr_scheduler_class
