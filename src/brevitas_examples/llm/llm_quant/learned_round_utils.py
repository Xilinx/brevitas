# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Any, Dict, List, Optional, Union

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from brevitas.utils.python_utils import recurse_getattr
from brevitas_examples.common.learned_round.learned_round_optimizer import Cache
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundOptimizer
from brevitas_examples.common.learned_round.learned_round_parser import parse_learned_round
from brevitas_examples.common.learned_round.learned_round_parser import \
    parse_learned_round_loss_class
from brevitas_examples.common.learned_round.learned_round_parser import parse_lr_scheduler_class
from brevitas_examples.common.learned_round.learned_round_parser import parse_optimizer_class


class CacheLLM(Cache, dict):

    def __init__(self) -> None:
        super().__init__()
        self.initialize_cache()

    def store_inputs(self, args, kwargs) -> None:
        self["args"].append(args)
        self["kwargs"].append(kwargs)

    def store_output(self, output) -> None:
        if isinstance(output, (tuple, list)):
            output = output[0]
        self["output"].append(output)

    def initialize_cache(self) -> None:
        self["args"] = []
        self["kwargs"] = []
        self["output"] = []

    def clear_cache(self) -> None:
        del self["args"]
        del self["kwargs"]
        del self["output"]
        self["args"] = []
        self["kwargs"] = []
        self["output"] = []

    def sample_batch(self, indices: torch.Tensor) -> Union[Any, torch.Tensor]:
        cache_args, cache_kwargs, cache_outs = self["args"], self["kwargs"], self["output"]
        # Positional arguments
        args = [cache_args[i] for i in indices]
        args = tuple(torch.cat(arg_tensor, dim=0) for arg_tensor in zip(*args))
        # Keyword arguments
        kwargs_dict = [cache_kwargs[i] for i in indices]
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
        outs = torch.cat([cache_outs[i] for i in indices], dim=0)
        return (args, kwargs), outs

    def __len__(self):
        return len(self["args"])


def llm_learned_round_prepare_fn(model: nn.Module) -> None:
    llm_cache_state = model.config.use_cache
    model.config.use_cache = False
    return llm_cache_state


def llm_learned_round_finish_fn(model: nn.Module, llm_cache_state: Dict) -> None:
    model.config.use_cache = llm_cache_state


def llm_forward(model: nn.Module, inputs: Any) -> None:
    device = next(model.parameters()).device
    if device != torch.device("meta"):
        inputs = send_to_device(inputs, device)
    model(**inputs)


def llm_block_forward(block: nn.Module, inputs: Any) -> torch.Tensor:
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


def apply_learned_round(
        model: nn.Module,
        calibration_loader: DataLoader,
        iters: int = 200,
        learned_round: str = "linear_round",
        learned_round_loss: str = "mse",
        block_name_attribute: str = "layers",
        optimizer: str = "sign_sgd",
        batch_size: int = 8,
        learn_scale: bool = False,
        use_best_model: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        loss_scaling_factor: float = 1000,
        lr_scheduler: Optional[str] = "linear",
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
        learned_round_loss_kwargs: Optional[Dict] = None,
        scale_optimizer_class: Optional[str] = None,
        scale_optimizer_kwargs: Optional[Dict] = None,
        fast_update: bool = False) -> None:
    # Parse strings to obtain the arguments for the optimizer
    learned_round = parse_learned_round(learned_round)
    learned_round_loss_class = parse_learned_round_loss_class(learned_round_loss)
    optimizer_class = parse_optimizer_class(optimizer)
    scale_optimizer_class = parse_optimizer_class(scale_optimizer_class)
    lr_scheduler_class = parse_lr_scheduler_class(lr_scheduler)

    llm_block_check_fn = functools.partial(get_blocks, block_name_attribute=block_name_attribute)

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
        learn_scale=learn_scale,
        use_best_model=use_best_model,
        amp_dtype=amp_dtype,
        loss_scaling_factor=loss_scaling_factor,
        learned_round_loss_kwargs=learned_round_loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        scale_optimizer_kwargs=scale_optimizer_kwargs,
        scale_optimizer_class=scale_optimizer_class)
    cache = CacheLLM()
    learned_round_optimizer.apply_learned_round(
        model=model,
        model_forward=llm_forward,
        block_forward=llm_block_forward,
        data_loader=calibration_loader,
        cache=cache,
        get_blocks_fn=llm_block_check_fn,
        model_prepare_fn=llm_learned_round_prepare_fn,
        model_finish_fn=llm_learned_round_finish_fn,
        keep_gpu=False,
        fast_update=fast_update)
