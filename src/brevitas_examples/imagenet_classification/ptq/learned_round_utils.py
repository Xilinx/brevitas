# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Part of this code has been re-adapted from https://github.com/yhhhli/BRECQ
# under the following LICENSE:

# MIT License

# Copyright (c) 2021 Yuhang Li

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import functools
import re
from typing import Any, Callable, Dict, Optional, Tuple, Union
import warnings

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from brevitas import config
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.quant_tensor import QuantTensor
from brevitas_examples.common.learned_round.learned_round_optimizer import Cache
from brevitas_examples.common.learned_round.learned_round_optimizer import get_blocks
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundOptimizer
from brevitas_examples.common.learned_round.learned_round_parser import parse_learned_round
from brevitas_examples.common.learned_round.learned_round_parser import \
    parse_learned_round_loss_class
from brevitas_examples.common.learned_round.learned_round_parser import parse_lr_scheduler_class
from brevitas_examples.common.learned_round.learned_round_parser import parse_optimizer_class

config.IGNORE_MISSING_KEYS = True


def is_block(module: nn.Module, module_name: str, reg_exp: str = r"layer\d+") -> bool:
    return (re.search(reg_exp, module_name) is not None)


def is_layer(module: nn.Module, module_name: str) -> bool:
    return isinstance(module, QuantWBIOL)


BLOCK_CHECK_MAP = {
    "layerwise": is_layer,
    "blockwise": is_block,}


class CacheVision(Cache, dict):

    def __init__(self) -> None:
        super().__init__()
        self.batch_dim = 0
        self.initialize_cache()

    def store_inputs(self, args, kwargs) -> None:
        input_batch = args[0]
        if isinstance(input_batch, QuantTensor):
            input_batch = input_batch.value

        if hasattr(input_batch, 'names') and 'N' in input_batch.names:
            self.batch_dim = input_batch.names.index('N')
            input_batch.rename_(None)
            input_batch = input_batch.transpose(0, self.batch_dim)

        self["inputs"].append(input_batch)

    def store_output(self, output) -> None:
        if self.batch_dim is not None:
            output.rename_(None)
            output = output.transpose(0, self.batch_dim)

        self["output"].append(output)

    def initialize_cache(self) -> None:
        self["inputs"] = []
        self["output"] = []

    def clear_cache(self) -> None:
        del self["inputs"]
        del self["output"]
        self["inputs"] = []
        self["output"] = []

    def sample_batch(self, indices: torch.Tensor) -> Union[Any, torch.Tensor]:
        if isinstance(self["inputs"], list):
            self["inputs"] = torch.cat(self["inputs"], dim=self.batch_dim)
        if isinstance(self["output"], list):
            self["output"] = torch.cat(self["output"], dim=self.batch_dim)

        return self["inputs"][indices], self["output"][indices]

    def __len__(self):
        return (
            len(self["inputs"])
            if isinstance(self["inputs"], list) else self["inputs"].shape[self.batch_dim])


def vision_forward(model: nn.Module, inputs: Any) -> None:
    device = next(model.parameters()).device
    img, _ = inputs
    img = send_to_device(img, device)
    model(img)


def vision_block_forward(block: nn.Module, inputs: Any) -> torch.Tensor:
    device = next(block.parameters()).device
    inputs = send_to_device(inputs, device)
    return block(inputs)


def apply_learned_round(
    model: nn.Module,
    calibration_loader: DataLoader,
    iters: int = 1000,
    learned_round: str = "hard_sigmoid_round",
    learned_round_loss: str = "regularised_mse",
    block_name_attribute: str = r"layer\d+",
    optimizer: str = "adam",
    lr_scheduler: Optional[str] = None,
    optimizer_lr: float = 1e-3,
    batch_size: int = 1,
    use_best_model: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    loss_scaling_factor: float = 1.,
    learned_round_loss_kwargs: Optional[Dict] = None,
    optimizer_kwargs: Optional[Dict] = None,
    lr_scheduler_kwargs: Optional[Dict] = None,
    learned_round_mode: str = "layerwise",
) -> None:
    # Parse strings to obtain the arguments for the optimizer
    learned_round = parse_learned_round(learned_round)
    learned_round_loss_class = parse_learned_round_loss_class(learned_round_loss)
    optimizer_class = parse_optimizer_class(optimizer)
    lr_scheduler_class = parse_lr_scheduler_class(lr_scheduler)

    # Parse method to retrieve de model blocks
    if learned_round_mode == "layerwise":
        block_check_fn = is_layer
    elif learned_round_mode == "blockwise":
        block_check_fn = functools.partial(is_block, reg_exp=block_name_attribute)
    else:
        block_check_fn = is_layer
        warnings.warn(
            f"{learned_round_mode} is not a valid learned round mode. Defaulting to layerwise.")
    get_blocks_fn = functools.partial(get_blocks, block_check_fn=block_check_fn)

    lr_scheduler_kwargs = {
        "start_factor": 1.0,
        "end_factor": 0.0,
        "verbose": False,} if lr_scheduler_kwargs is None else lr_scheduler_kwargs
    learned_round_optimizer = LearnedRoundOptimizer(
        learned_round=learned_round,
        learned_round_loss_class=learned_round_loss_class,
        optimizer_class=optimizer_class,
        lr_scheduler_class=lr_scheduler_class,
        batch_size=batch_size,
        iters=iters,
        use_best_model=use_best_model,
        amp_dtype=amp_dtype,
        loss_scaling_factor=loss_scaling_factor,
        learned_round_loss_kwargs=learned_round_loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_kwargs=lr_scheduler_kwargs)
    cache = CacheVision()
    learned_round_optimizer.apply_learned_round(
        model=model,
        model_forward=vision_forward,
        block_forward=vision_block_forward,
        data_loader=calibration_loader,
        cache=cache,
        get_blocks_fn=get_blocks_fn,
        keep_gpu=True,
    )
