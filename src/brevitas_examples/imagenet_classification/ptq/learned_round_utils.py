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

from argparse import Namespace
import functools
import re
from typing import Any
from typing import List
import warnings

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from brevitas import config
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.quant_tensor import QuantTensor
from brevitas_examples.common.learned_round.learned_round_args import HandlerSpec
from brevitas_examples.common.learned_round.learned_round_args import LossArgs
from brevitas_examples.common.learned_round.learned_round_args import LRSchedulerArgs
from brevitas_examples.common.learned_round.learned_round_args import OptimizerArgs
from brevitas_examples.common.learned_round.learned_round_args import TrainerConfig
from brevitas_examples.common.learned_round.learned_round_args import TrainingArgs
from brevitas_examples.common.learned_round.learned_round_method import LearnedRoundArgs
from brevitas_examples.common.learned_round.learned_round_trainer import Cache
from brevitas_examples.common.learned_round.learned_round_trainer import get_blocks
from brevitas_examples.common.learned_round.learned_round_trainer import LearnedRoundTrainer

config.IGNORE_MISSING_KEYS = True


def is_block(module: nn.Module, module_name: str, reg_exp: str = r"layer\d+") -> bool:
    return (re.search(reg_exp, module_name) is not None)


def is_layer(module: nn.Module, module_name: str) -> bool:
    return isinstance(module, QuantWBIOL)


class CacheVision(Cache[torch.Tensor, torch.Tensor]):

    def __init__(self) -> None:
        self.batch_dim = 0
        self.inputs: List[torch.Tensor] = []
        self.outputs: List[torch.Tensor] = []

    def store_inputs(self, args, kwargs) -> None:
        input_batch = args[0]
        if isinstance(input_batch, QuantTensor):
            input_batch = input_batch.value

        if hasattr(input_batch, 'names') and 'N' in input_batch.names:
            self.batch_dim = input_batch.names.index('N')
            input_batch.rename_(None)
            input_batch = input_batch.transpose(0, self.batch_dim)

        self.inputs.extend(torch.split(input_batch, 1, self.batch_dim))

    def store_output(self, output) -> None:
        if self.batch_dim is not None:
            output.rename_(None)
            output = output.transpose(0, self.batch_dim)

        self.outputs.extend(torch.split(output, 1, 0))

    def reset_cache(self) -> None:
        self.inputs = []
        self.outputs = []

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    def collate_fn(self, batch):
        inps, outs = zip(*batch)
        # Concatenate across the batch dimension
        inps = torch.cat(inps, dim=self.batch_dim)
        outs = torch.cat(outs, dim=0)
        return inps, outs


def vision_forward(model: nn.Module, inputs: Any) -> None:
    device = next(model.parameters()).device
    img, _ = inputs
    img = send_to_device(img, device)
    model(img)


def vision_block_forward(block: nn.Module, inputs: Any) -> torch.Tensor:
    device = next(block.parameters()).device
    inputs = send_to_device(inputs, device)
    return block(inputs)


# TODO (pml): Transition to `args` being a nested dictionary, which is translated
# an validated to `Config`` (e.g. using the package dacite)
def parse_args_to_dataclass(args: Namespace) -> TrainerConfig:
    lr_scheduler_args = None
    if args.learned_round_lr_scheduler is not None:
        lr_scheduler_args = LRSchedulerArgs(
            lr_scheduler_cls=args.learned_round_lr_scheduler,
            lr_scheduler_kwargs={
                "start_factor": 1.0,
                "end_factor": 0.0,
                "total_iters": args.learned_round_iters,},
        )

    optim_args = OptimizerArgs(
        target_params="learned_round",
        optimizer_cls="Adam",
        lr=args.learned_round_lr,
        optimizer_kwargs={},
        lr_scheduler_args=lr_scheduler_args,
    )

    if args.learned_round_loss == "regularised_mse":
        losses_args = [
            LossArgs(
                cls="round_reg",
                kwargs=None,
            ),
            LossArgs(
                cls="mse",
                kwargs={
                    "reduction": "none",
                    "dim": 1,},
            )]
    elif args.learned_round_loss == "mse":
        losses_args = [LossArgs(cls="mse",)]
    else:
        raise ValueError(f"{args.learned_round_loss} is not a valid learned round loss.")

    training_args = TrainingArgs(
        optimizers_args=[optim_args],
        batch_size=args.learned_round_batch_size,
        iters=args.learned_round_iters,
        losses_args=losses_args,
        loss_scaling_factor=1.0,
        use_best_model=False,
        use_amp=True,
        amp_dtype="float16",
        fast_update=False,
    )

    learned_round_args = LearnedRoundArgs(
        learned_round_param=args.learned_round,
        learned_round_kwargs=None,
    )

    training_handlers = [
        HandlerSpec[LearnedRoundArgs](
            name="learned_round",
            config=learned_round_args,
        )]

    return TrainerConfig(
        training_args=training_args,
        training_handlers=training_handlers,
    )


def apply_learned_round(
        model: nn.Module,
        calibration_loader: DataLoader,
        args: Namespace,
        block_name_attribute: str = r"layer\d+") -> None:
    # Instantiate cache for vision models
    cache = CacheVision()
    # Parse method to retrieve de model blocks
    if args.learned_round_mode == "layerwise":
        block_check_fn = is_layer
    elif args.learned_round_mode == "blockwise":
        block_check_fn = functools.partial(is_block, reg_exp=block_name_attribute)
    else:
        block_check_fn = is_layer
        warnings.warn(
            f"{args.learned_round_mode} is not a valid learned round mode. Defaulting to layerwise."
        )
    get_blocks_fn = functools.partial(get_blocks, block_check_fn=block_check_fn)

    config = parse_args_to_dataclass(args)
    learned_round_trainer = LearnedRoundTrainer(config=config)
    learned_round_trainer.train(
        model=model,
        model_forward=vision_forward,
        block_forward=vision_block_forward,
        data_loader=calibration_loader,
        cache=cache,
        get_blocks_fn=get_blocks_fn,
        keep_gpu=True)
