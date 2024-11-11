# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, List, Tuple, Union

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data.dataloader import DataLoader
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from brevitas.optim.sign_sgd import SignSGD
from brevitas_examples.common.learned_round.learned_round_method import AutoRound
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundOptimizer


class CacheLLM(dict):

    def __init__(self) -> None:
        super().__init__()
        self.store_kwargs = True

    def store_inputs(self, args, kwargs) -> None:
        self["args"].append(args)
        if self.store_kwargs:
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
        del self["output"]
        self["args"] = []
        self["output"] = []
        self.store_kwargs = len(self["kwargs"]) == 0

    def reset_cache(self) -> None:
        del self["args"]
        del self["kwargs"]
        del self["output"]
        self.store_kwargs = True
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


def llm_block_check_fn(module: nn.Module, module_name: str) -> bool:
    return isinstance(module, LlamaDecoderLayer) or isinstance(module, OPTDecoderLayer)


def apply_learned_round(model: nn.Module, calibration_loader: DataLoader) -> None:
    iters = 200
    learned_round = AutoRound(iters=200)
    optimizer_class = SignSGD
    lr_scheduler_class = LinearLR
    optimizer_lr = 5e-3
    batch_size = 8
    use_best_model = True
    use_amp = True
    amp_dtype = torch.float16
    loss_scaling_factor = 1000.
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
    cache = CacheLLM()
    learned_round_optimizer.apply_learned_round(
        model=model,
        model_forward=llm_forward,
        block_forward=llm_block_forward,
        data_loader=calibration_loader,
        cache=cache,
        block_check_fn=llm_block_check_fn,
        model_prepare_fn=llm_learned_round_prepare_fn,
        model_finish_fn=llm_learned_round_finish_fn,
        keep_gpu=True,
    )
