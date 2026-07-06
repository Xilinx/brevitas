# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import functools
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from transformers import PreTrainedModel

from brevitas.utils.python_utils import recurse_getattr
from brevitas_examples.common.learned_round.learned_round_args import HandlerSpec
from brevitas_examples.common.learned_round.learned_round_args import LossArgs
from brevitas_examples.common.learned_round.learned_round_args import LRSchedulerArgs
from brevitas_examples.common.learned_round.learned_round_args import OptimizerArgs
from brevitas_examples.common.learned_round.learned_round_args import TrainerConfig
from brevitas_examples.common.learned_round.learned_round_args import TrainingArgs
from brevitas_examples.common.learned_round.learned_round_method import LearnedRoundArgs
from brevitas_examples.common.learned_round.learned_round_trainer import Cache
from brevitas_examples.common.learned_round.learned_round_trainer import LearnedRoundTrainer

T_args = Tuple[torch.Tensor, ...]
T_kwargs = Dict[str, Any]
T_inputs = Tuple[T_args, T_kwargs]
T_outputs = torch.Tensor


class CacheLLM(Cache[T_inputs, T_outputs]):

    def __init__(self) -> None:
        self._args: List[T_args] = []
        self._kwargs: List[T_kwargs] = []
        self.outputs: List[T_outputs] = []

    def store_inputs(self, args, kwargs):
        args = list(zip(*map(lambda x: list(torch.split(x, 1, dim=0)), args)))
        self._args.extend(args)
        bs = len(args)
        kwargs_split = {
            key:
            value if not isinstance(value, torch.Tensor) else list(torch.split(value, 1, dim=0))
            for key,
            value in kwargs.items()}
        kwargs = [{
            key: value if not isinstance(value, list) else value[i] for key,
            value in kwargs_split.items()} for i in range(bs)]
        self._kwargs.extend(kwargs)

    def store_output(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = list(torch.split(output, 1, dim=0))
        self.outputs.extend(output)

    def reset_cache(self) -> None:
        self._args = []
        self._kwargs = []
        self.outputs = []

    def __len__(self):
        return len(self._args)

    def __getitem__(self, index):
        return (self._args[index], self._kwargs[index]), self.outputs[index]

    def collate_fn(self, batch):
        inps, outs = zip(*batch)
        args, kwargs_dict = zip(*inps)
        # Positional arguments
        args = tuple(torch.cat(arg_tensor, dim=0) for arg_tensor in zip(*args))
        # Keyword arguments
        kwargs = {}
        for curr_dict in kwargs_dict:
            for key, value in curr_dict.items():
                if isinstance(value, torch.Tensor):
                    if key not in kwargs:
                        kwargs[key] = []
                    kwargs[key].append(value)
                else:
                    if key not in kwargs:
                        kwargs[key] = value
        for key, value in kwargs.items():
            if isinstance(value, list) and len(value) > 0:
                kwargs[key] = torch.cat(kwargs[key], dim=0)
        # FP outputs
        outs = torch.cat(outs, dim=0)
        return (args, kwargs), outs

    @property
    def inputs(self):
        return (self._args, self._kwargs)

    @inputs.setter
    def inputs(self, new_inputs):
        if not isinstance(new_inputs, tuple):
            # If only args were passed, verify that each element is a tuple
            new_args = list(map(lambda arg: arg if isinstance(arg, tuple) else (arg,), new_inputs))
            new_inputs = (new_args, self._kwargs)
        # Update the inputs of the cache
        self._args, self._kwargs = new_inputs

    # Auxiliar functions to perform fast_update
    def collate_fn_output_next(self, batch):
        (_, kwargs), outputs = self.collate_fn(batch)
        return (outputs,), kwargs

    def collate_fn_input_next(self, batch):
        (args, kwargs), _ = self.collate_fn(batch)
        return args, kwargs


def llm_forward(model: nn.Module, inputs: Dict[str, Any]) -> None:
    device = next(model.parameters()).device
    if device != torch.device("meta"):
        inputs = send_to_device(inputs, device)
    model(**inputs)


def llm_block_forward(block: nn.Module, inputs: T_inputs) -> torch.Tensor:
    device = next(block.parameters()).device
    args, kwargs = inputs
    args = send_to_device(args, device)
    kwargs = send_to_device(kwargs, device)
    out = block(*args, **kwargs)
    if isinstance(out, tuple):
        out = out[0]
    return out


def get_blocks(model: nn.Module, block_name_attribute: str) -> List[nn.Module]:
    return recurse_getattr(model, block_name_attribute)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Keyword arguments
    kwargs = {}
    for curr_dict in batch:
        for key, value in curr_dict.items():
            if isinstance(value, torch.Tensor):
                if key not in kwargs:
                    kwargs[key] = []
                kwargs[key].append(value)
            else:
                if key not in kwargs:
                    kwargs[key] = value
    for key, value in kwargs.items():
        if isinstance(value, list) and len(value) > 0:
            kwargs[key] = torch.cat(kwargs[key], dim=0)
    return kwargs


# TODO (pml): Transition to `args` being a nested dictionary, which is translated
# an validated to `Config`` (e.g. using the package dacite)
def parse_args_to_dataclass(args: Namespace) -> TrainerConfig:

    def _parse_lr_scheduler_args(args: Namespace) -> LRSchedulerArgs:
        return LRSchedulerArgs(
            lr_scheduler_cls="LinearLR",
            lr_scheduler_kwargs={
                "start_factor": 1.0,
                "end_factor": 0.0,
                "total_iters": args.learned_round_iters,},
        )

    # Optimizer for learned round parameters
    learned_round_optim_args = OptimizerArgs(
        target_params="learned_round",
        optimizer_cls="SignSGD",
        lr=args.learned_round_lr,
        optimizer_kwargs={},
        lr_scheduler_args=_parse_lr_scheduler_args(args),
    )

    # Optimizer for scales
    scales_optim_args = OptimizerArgs(
        target_params="scales",
        optimizer_cls="SGD",
        lr=args.learned_round_scale_lr,
        optimizer_kwargs={
            "momentum": args.learned_round_scale_momentum,},
        lr_scheduler_args=_parse_lr_scheduler_args(args),
    )

    training_args = TrainingArgs(
        optimizers_args=[learned_round_optim_args] +
        ([scales_optim_args] if args.learned_round_scale else []),
        batch_size=8,
        iters=args.learned_round_iters,
        losses_args=[
            LossArgs(cls="mse",),],
        loss_scaling_factor=1000.0,
        use_best_model=True,
        use_amp=True,
        amp_dtype="float16",
        fast_update=args.learned_round_fast_update,
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


class llm_trainer_cm:

    def __init__(self, model: PreTrainedModel) -> None:
        self.model = model
        self.model_cache_state = None

    def __enter__(self) -> None:
        self.model_cache_state = self.model.config.use_cache
        self.model.config.use_cache = False

    def __exit__(self, type, value, traceback) -> None:
        self.model.config.use_cache = self.model_cache_state


def apply_learned_round(
        model: nn.Module, calibration_loader: torch.utils.data.DataLoader, args: Namespace) -> None:
    cache = CacheLLM()
    llm_block_check_fn = functools.partial(get_blocks, block_name_attribute=args.gpxq_block_name)

    config = parse_args_to_dataclass(args)
    learned_round_trainer = LearnedRoundTrainer(config=config)
    with llm_trainer_cm(model):
        learned_round_trainer.train(
            model=model,
            model_forward=llm_forward,
            block_forward=llm_block_forward,
            data_loader=calibration_loader,
            cache=cache,
            get_blocks_fn=llm_block_check_fn,
            keep_gpu=False)
