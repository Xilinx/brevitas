# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from accelerate.utils.operations import send_to_device
from datasets import Dataset
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from brevitas.optim.sign_sgd import SignSGD
from brevitas_examples.common.learned_round.learned_round_method import AutoRound
from brevitas_examples.common.learned_round.learned_round_method import LearnedRound
from brevitas_examples.common.learned_round.learned_round_method import LearnedRoundLoss
from brevitas_examples.common.learned_round.learned_round_method import MSELoss
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundOptimizer

LEARNED_ROUND_MAP = {
    "auto_round": AutoRound,}
LEARNED_ROUND_LOSS_MAP = {
    "mse": MSELoss,}
OPTIMIZER_MAP = {
    "sign_sgd": SignSGD,}
LR_SCHEDULER_MAP = {
    "linear": LinearLR,}


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

    def cache_to_dataset(self) -> Dataset:
        inputs_list = list(zip(self["args"], self["kwargs"]))
        return list(zip(inputs_list, self["output"]))

    def collate_fn(self, batch: Any) -> Any:
        # Format of the dataset is ((args, kwargs), outs)
        # See cache_to_dataset
        inputs, outs = map(list, zip(*batch))
        args, kwargs_dict = map(list, zip(*inputs))
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
        return ((args, kwargs), outs)

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


def llm_block_check_fn(module: nn.Module, module_name: str) -> bool:
    return isinstance(module, LlamaDecoderLayer) or isinstance(module, OPTDecoderLayer)


def apply_learned_round(
    model: nn.Module,
    calibration_loader: DataLoader,
    iters: int = 200,
    learned_round: str = "auto_round",
    learned_round_loss: str = "mse",
    optimizer: str = "sign_sgd",
    lr_scheduler: Optional[str] = "linear",
    optimizer_lr: float = 5e-3,
    batch_size: int = 8,
    learn_scale: bool = False,
    use_best_model: bool = True,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    loss_scaling_factor: float = 1000,
    optimizer_kwargs: Optional[Dict] = None,
    lr_scheduler_kwargs: Optional[Dict] = None,
    learned_round_loss_kwargs: Optional[Dict] = None,
) -> None:
    if learned_round not in LEARNED_ROUND_MAP:
        raise ValueError(f"Learned round method {learned_round} is not available.")
    learned_round = LEARNED_ROUND_MAP[learned_round]()

    if learned_round_loss not in LEARNED_ROUND_LOSS_MAP:
        raise ValueError(f"Learned round loss {learned_round_loss} is not available.")
    learned_round_loss_class = LEARNED_ROUND_LOSS_MAP[learned_round_loss]

    if optimizer not in OPTIMIZER_MAP:
        raise ValueError(f"Optimizer {optimizer} is not available.")
    optimizer_class = OPTIMIZER_MAP[optimizer]

    if lr_scheduler is not None and lr_scheduler not in LR_SCHEDULER_MAP:
        raise ValueError(f"Learning rate scheduler {lr_scheduler} is not available.")
    lr_scheduler_class = None if lr_scheduler is None else LR_SCHEDULER_MAP[lr_scheduler]

    lr_scheduler_kwargs = {
        "start_factor": 1.0,
        "end_factor": 0.0,
        "verbose": False,} if lr_scheduler_kwargs is None else lr_scheduler_kwargs
    learned_round_optimizer = LearnedRoundOptimizer(
        learned_round=learned_round,
        learned_round_loss_class=learned_round_loss_class,
        optimizer_class=optimizer_class,
        lr_scheduler_class=lr_scheduler_class,
        optimizer_lr=optimizer_lr,
        batch_size=batch_size,
        iters=iters,
        learn_scale=learn_scale,
        use_best_model=use_best_model,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        loss_scaling_factor=loss_scaling_factor,
        learned_round_loss_kwargs=learned_round_loss_kwargs,
        optimizer_kwargs=optimizer_kwargs,
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
        keep_gpu=False,
    )
