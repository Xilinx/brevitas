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

import re
from typing import Any, Callable, Tuple, Union
import warnings

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from brevitas import config
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.optim.sign_sgd import SignSGD
from brevitas.quant_tensor import QuantTensor
from brevitas_examples.common.learned_round.learned_round_method import AdaRound
from brevitas_examples.common.learned_round.learned_round_method import AutoRound
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundOptimizer

config.IGNORE_MISSING_KEYS = True


class CacheCNN(dict):

    def __init__(self) -> None:
        super().__init__()
        self.batch_dim = 0

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

    def reset_cache(self) -> None:
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


def cnn_forward(model: nn.Module, inputs: Any) -> None:
    device = next(model.parameters()).device
    img, _ = inputs
    img = send_to_device(img, device)
    model(img)


def cnn_block_forward(block: nn.Module, inputs: Any) -> torch.Tensor:
    device = next(block.parameters()).device
    inputs = send_to_device(inputs, device)
    return block(inputs)


def is_resnet_block(module: nn.Module, module_name: str) -> bool:
    return (re.search(r"layer\d+", module_name) is not None)


def is_layer(module: nn.Module, module_name: str) -> bool:
    return isinstance(module, QuantWBIOL)


def apply_learned_round(
    model: nn.Module,
    calibration_loader: DataLoader,
    learned_round_name: str = "ada_round",
    optimizer: str = "adam",
    learned_round_mode: str = "layerwise",
    iters: int = 1000,
    optimizer_lr: float = 1e-3,
    batch_size: int = 1,
) -> None:
    optimizer_classes = {"adam": torch.optim.Adam, "sign_sgd": SignSGD}
    if optimizer not in optimizer_classes:
        raise ValueError(f"{optimizer} is not a valid optimizer.")
    optimizer_class = optimizer_classes[optimizer]

    block_check_fns = {"layerwise": is_layer, "blockwise": is_resnet_block}
    if learned_round_mode not in block_check_fns:
        learned_round_mode = "layerwise"
        warnings.warn(
            f"{learned_round_mode} is not a valid learned round mode. Defaulting to layerwise.")
    block_check_fn = block_check_fns[learned_round_mode]

    learned_round_methods = {"ada_round": AdaRound, "auto_round": AutoRound}
    if learned_round_name not in learned_round_methods:
        raise ValueError(f"Learned round method {learned_round_name} is not available.")
    learned_round = learned_round_methods[learned_round_name](iters=iters)

    lr_scheduler_class = None if optimizer == "adam" else torch.optim.lr_scheduler.LinearLR
    use_best_model = False if learned_round_name == "ada_round" else True
    use_amp = True
    amp_dtype = torch.float16
    loss_scaling_factor = 1.
    optimizer_kwargs = None
    lr_scheduler_kwargs = {
        "start_factor": 1.0,
        "end_factor": 0.0,
        "verbose": False,}
    learned_round_optimizer = LearnedRoundOptimizer(
        learned_round=learned_round,
        optimizer_class=optimizer_class,
        lr_scheduler_class=lr_scheduler_class,
        optimizer_lr=optimizer_lr,
        batch_size=batch_size,
        iters=iters,
        use_best_model=use_best_model,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        loss_scaling_factor=loss_scaling_factor,
        optimizer_kwargs={} if optimizer_kwargs is None else optimizer_kwargs,
        lr_scheduler_kwargs=lr_scheduler_kwargs)
    cache = CacheCNN()
    learned_round_optimizer.apply_learned_round(
        model=model,
        model_forward=cnn_forward,
        block_forward=cnn_block_forward,
        data_loader=calibration_loader,
        cache=cache,
        block_check_fn=block_check_fn,
        keep_gpu=True,
    )