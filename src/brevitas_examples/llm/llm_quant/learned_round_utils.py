# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import functools
from typing import Any, Dict, List, Tuple

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from brevitas.utils.python_utils import recurse_getattr
from brevitas_examples.common.learned_round.learned_round_args import Config
from brevitas_examples.common.learned_round.learned_round_optimizer import Cache
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundOptimizer

_T_args = Tuple[torch.Tensor, ...]
_T_kwargs = Dict[str, Any]
_T_inputs = Tuple[_T_args, _T_kwargs]
_T_outputs = torch.Tensor


class CacheLLM(Cache[_T_inputs, _T_outputs]):

    def __init__(self) -> None:
        self._args: List[_T_args] = []
        self._kwargs: List[_T_kwargs] = []
        self.outputs: List[_T_outputs] = []

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


def llm_learned_round_prepare_fn(model: nn.Module) -> None:
    llm_cache_state = model.config.use_cache
    model.config.use_cache = False
    return llm_cache_state


def llm_learned_round_finish_fn(model: nn.Module, llm_cache_state: bool) -> None:
    model.config.use_cache = llm_cache_state


def llm_forward(model: nn.Module, inputs: Dict[str, Any]) -> None:
    device = next(model.parameters()).device
    if device != torch.device("meta"):
        inputs = send_to_device(inputs, device)
    model(**inputs)


def llm_block_forward(block: nn.Module, inputs: _T_inputs) -> torch.Tensor:
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


def parse_args_to_dataclass(args: Namespace) -> Config:
    # TODO: Remove, only kept for retrocompatibility
    from brevitas.inject.enum import LearnedRoundImplType
    LEARNED_ROUND_MAP = {
        "linear_round": LearnedRoundImplType.IDENTITY,
        "hard_sigmoid_round": LearnedRoundImplType.HARD_SIGMOID,
        "sigmoid_round": LearnedRoundImplType.SIGMOID,}

    config_dict = {
        "learned_round_args": {
            # TODO: Remove, only used to map to new names
            "learned_round_param": LEARNED_ROUND_MAP[args.learned_round].value.lower(),
            "learned_round_kwargs": None,
            "loss_cls": "mse",
            "loss_kwargs": None,
            "fast_update": args.learned_round_fast_update,},
        "training_args": {
            "optimizers_args": [
                {
                    "optimizer_cls": "sign_sgd",
                    "lr": args.learned_round_lr,
                    "optimizer_kwargs": {},
                    "lr_scheduler_args": {
                        "lr_scheduler_cls":
                            "linear",
                        "lr_scheduler_kwargs":
                            f'{{"start_factor": 1.0, "end_factor": 0.0, "total_iters": {args.learned_round_iters}}}'
                    }},
                {
                    "optimizer_cls": "sgd",
                    "lr": args.learned_round_scale_lr,
                    "optimizer_kwargs": {
                        "momentum": args.learned_round_scale_momentum,},
                    "lr_scheduler_args": {
                        "lr_scheduler_cls":
                            "linear",
                        "lr_scheduler_kwargs":
                            f'{{"start_factor": 1.0, "end_factor": 0.0, "total_iters": {args.learned_round_iters}}}'
                    }}],
            "block_name_attribute":
                args.gpxq_block_name,
            "optimizers_targets": ["learned_round"] +
                                  (["scales"] if args.learned_round_scale else []),
            "batch_size":
                8,
            "iters":
                args.learned_round_iters,
            "use_best_model":
                True,
            "use_amp":
                True,
            "amp_dtype":
                "float16",}}
    from dacite import from_dict
    config = from_dict(data_class=Config, data=config_dict)
    return config


def apply_learned_round(model: nn.Module, calibration_loader: DataLoader, args: Namespace) -> None:
    cache = CacheLLM()
    llm_block_check_fn = functools.partial(get_blocks, block_name_attribute=args.gpxq_block_name)

    config = parse_args_to_dataclass(args)
    learned_round_optimizer = LearnedRoundOptimizer(config=config)
    learned_round_optimizer.apply_learned_round(
        model=model,
        model_forward=llm_forward,
        block_forward=llm_block_forward,
        dataset=calibration_loader,
        cache=cache,
        get_blocks_fn=llm_block_check_fn,
        collate_fn=collate_fn,
        model_prepare_fn=llm_learned_round_prepare_fn,
        model_finish_fn=llm_learned_round_finish_fn,
        keep_gpu=False)
