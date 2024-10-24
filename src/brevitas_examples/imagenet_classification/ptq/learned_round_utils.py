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

from abc import ABC
from abc import abstractmethod
from typing import Callable, Generator, List, Optional, Tuple

from accelerate.utils.operations import send_to_device
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from brevitas import config
from brevitas.core.function_wrapper.auto_round import AutoRoundSte
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.graph.calibrate import restore_return_quant_tensor
from brevitas.inject.enum import FloatToIntImplType
from brevitas.inject.enum import LearnedRoundImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.quant_tensor import QuantTensor

config.IGNORE_MISSING_KEYS = True


class StopFwdException(Exception):
    """Used to throw and catch an exception to stop traversing the graph."""
    pass


class DataSaverHook:

    def __init__(self, store_output: False):
        self.store_output = store_output
        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        input_batch = input_batch[0]
        if isinstance(input_batch, QuantTensor):
            input_batch = input_batch.value

        if hasattr(input_batch, 'names') and 'N' in input_batch.names:
            batch_dim = input_batch.names.index('N')

            input_batch.rename_(None)
            input_batch = input_batch.transpose(0, batch_dim)
            if self.store_output:
                output_batch.rename_(None)
                output_batch = output_batch.transpose(0, batch_dim)

        if self.store_output:
            self.output_store = output_batch
        self.input_store = input_batch
        raise StopFwdException


class LinearTempDecay:

    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


class LearnedRoundLoss(ABC):

    @abstractmethod
    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        pass

    @abstractmethod
    def format_loss_components(self, *args) -> str:
        pass


class AdaRoundLoss(LearnedRoundLoss):

    def __init__(
            self,
            module: nn.Module,
            learned_round_modules: List[nn.Module],
            weight: float = 0.01,
            max_count: int = 1000,
            b_range: Tuple = (20, 2),
            warmup: float = 0.2,
            decay_start: float = 0.0) -> None:
        super().__init__()
        # AdaRound operates in a layer-wise manner, so integrity needs to be checked
        assert isinstance(module, QuantWBIOL), "AdaRound can only accept a single QuantWBIOL layer."
        assert len(learned_round_modules) == 1, "AdaRound can only accept a single learned round module."

        self.weight = weight
        self.module = module
        self.loss_start = max_count * warmup
        self.temp_decay = LinearTempDecay(
            max_count,
            start_b=b_range[0],
            end_b=b_range[1],
            rel_start_decay=warmup + (1.0 - warmup) * decay_start)
        self.iter = 0
        self.learned_round_module = learned_round_modules[0]

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        self.iter += 1

        rec_loss = F.mse_loss(pred, tgt, reduction='none').sum(1).mean()

        if self.iter < self.loss_start:
            b = self.temp_decay(self.iter)
            round_loss = 0
        else:  # 1 - |(h-0.5)*2|**b
            b = self.temp_decay(self.iter)
            round_vals = self.learned_round_module.p_forward()
            round_loss = self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()

        total_loss = rec_loss + round_loss
        return total_loss, (total_loss, rec_loss, round_loss, b)

    def format_loss_components(self, loss: float, rec_loss: float, round_loss: float, b) -> str:
        return "loss = {:.4f}, rec_loss = {:.4f}, round_loss = {:.4f}, b = {:.4f}".format(
            loss, rec_loss, round_loss, b)


class AutoRoundLoss(LearnedRoundLoss):

    def __call__(self, pred: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        loss = F.mse_loss(pred, tgt, reduction='none').sum(1).mean()
        return loss, (loss,)

    def format_loss_components(self, loss: float) -> str:
        return "loss = {:.4f}".format(loss)


class LearnedRound(ABC):

    def __init__(self, iters: int = 100) -> None:
        self.iters = iters

    def _insert_learned_round_quantizer(self, block: nn.Module) -> None:
        for module in block.modules():
            if isinstance(module, QuantWBIOL) and len(
                    self._find_learned_round_modules(module)) == 0:
                self._insert_learned_round_quantizer_to_layer(module)
                module.weight_quant.init_tensor_quant(preserve_state_dict=True)

    @abstractmethod
    def _insert_learned_round_quantizer_to_layer(self, layer: nn.Module) -> None:
        pass

    @abstractmethod
    def _is_learned_round_module(self, module: nn.Module) -> bool:
        pass

    @abstractmethod
    def _instantiate_loss(
            self, block: nn.Module, learned_round_modules: List[nn.Module]) -> LearnedRoundLoss:
        pass

    def _find_learned_round_modules(self, block: nn.Module) -> List[nn.Module]:
        round_modules = []
        for module in block.modules():
            if self._is_learned_round_module(module):
                round_modules.append(module)
        return round_modules

    def learned_round_iterator(
            self,
            blocks: List[nn.Module]) -> Generator[nn.Module, LearnedRoundLoss, List[nn.Module]]:
        for block in blocks:
            # Insert learned round quantizers into the appropiate submodules
            self._insert_learned_round_quantizer(block)
            # Freeze block parameters
            for params in block.parameters():
                params.requires_grad = False
            # Retrieve learned round modules
            learned_round_modules = self._find_learned_round_modules(block)
            # Enable gradient tracking in learned round modules
            for round_module in learned_round_modules:
                for params in round_module.parameters():
                    params.requires_grad = True
            block_loss = self._instantiate_loss(block, learned_round_modules)
            yield block, block_loss, learned_round_modules
            block.eval()


class AdaRound(LearnedRound):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _is_learned_round_module(self, module: nn.Module) -> bool:
        return isinstance(module, LearnedRoundSte)

    def _insert_learned_round_quantizer_to_layer(
            self,
            layer: nn.Module,
            learned_round_zeta: float = 1.1,
            learned_round_gamma: float = -0.1) -> None:
        floor_weight = torch.floor(layer.weight.data / layer.quant_weight().scale)
        delta = (layer.weight.data / layer.quant_weight().scale) - floor_weight
        value = -torch.log((learned_round_zeta - learned_round_gamma) /
                           (delta - learned_round_gamma) - 1)
        layer.weight_quant.quant_injector = layer.weight_quant.quant_injector.let(
            float_to_int_impl_type=FloatToIntImplType.LEARNED_ROUND,
            learned_round_impl_type=LearnedRoundImplType.HARD_SIGMOID,
            learned_round_gamma=learned_round_gamma,
            learned_round_zeta=learned_round_zeta,
            learned_round_init=value)

    def _instantiate_loss(
            self, block: nn.Module, learned_round_modules: List[nn.Module]) -> AdaRoundLoss:
        return AdaRoundLoss(block, learned_round_modules, max_count=self.iters)


class AutoRound(LearnedRound):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _is_learned_round_module(self, module: nn.Module) -> bool:
        return isinstance(module, AutoRoundSte)

    def _insert_learned_round_quantizer_to_layer(self, layer: nn.Module) -> None:
        value = torch.zeros_like(layer.weight.data)
        layer.weight_quant.quant_injector = layer.weight_quant.quant_injector.let(
            float_to_int_impl_type=FloatToIntImplType.AUTO_ROUND,
            learned_round_init=value,
        )

    def _instantiate_loss(
            self, block: nn.Module, learned_round_modules: List[nn.Module]) -> AutoRoundLoss:
        return AutoRoundLoss()


def save_inp_out_data(
        model,
        module,
        dataloader: torch.utils.data.DataLoader,
        store_inp=False,
        store_out=False,
        keep_gpu: bool = True,
        disable_quant=False):
    if disable_quant:
        disable_quant_class = DisableEnableQuantization()
        disable_quant_class.disable_act_quantization(model, False)
        disable_quant_class.disable_param_quantization(model, False)
        return_quant_tensor_state = disable_return_quant_tensor(model)
    device = next(model.parameters()).device
    data_saver = DataSaverHook(store_output=store_out)
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.no_grad():
        for img, t in dataloader:
            try:
                _ = model(img.to(device))
            except StopFwdException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store.detach())
                else:
                    cached[0].append(data_saver.input_store.detach().cpu())
            if store_out:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    cached[1].append(data_saver.output_store.detach().cpu())
    if store_inp:
        cached[0] = torch.cat([x for x in cached[0]], dim=0)
    if store_out:
        cached[1] = torch.cat([x for x in cached[1]], dim=0)
    handle.remove()
    if disable_quant:
        disable_quant_class.enable_act_quantization(model, False)
        disable_quant_class.enable_param_quantization(model, False)
        restore_return_quant_tensor(model, return_quant_tensor_state)
    return cached


class DataSaverHookLLM:

    def __init__(
            self,
            cache_args: List,
            cache_kwargs: List,
            cache_outs: List,
            store_args: bool = True,
            store_kwargs: bool = True,
            store_outs: bool = True,
            keep_gpu: bool = True):
        self.cache_args = cache_args
        self.cache_kwargs = cache_kwargs
        self.cache_outs = cache_outs

        self.store_args = store_args
        self.store_kwargs = store_kwargs
        self.store_outs = store_outs

        self.keep_gpu = keep_gpu

    def __call__(self, module, args, kwargs, output):
        # NOTE: If args/kwargs are QuantTensors, should include logic to unpack their values
        if isinstance(output, (tuple, list)):
            output = output[0]

        # Store each element in the appropiate cache
        for element_to_cache, should_cache, cache in zip(
            [args, kwargs, output],
            [self.store_args, self.store_kwargs, self.store_outs],
            [self.cache_args, self.cache_kwargs, self.cache_outs]
        ):
            if should_cache:
                if not self.keep_gpu:
                    element_to_cache = send_to_device(element_to_cache, 'cpu')
                cache.append(element_to_cache)

        raise StopFwdException


def save_inp_out_data_llm(
        model,
        module,
        dataloader: torch.utils.data.DataLoader,
        cache_args: List,
        cache_kwargs: List,
        cache_outs: List,
        store_args: bool = True,
        store_kwargs: bool = False,
        store_outs: bool = True,
        keep_gpu: bool = True,
        disable_quant=False) -> None:
    if disable_quant:
        disable_quant_class = DisableEnableQuantization()
        disable_quant_class.disable_act_quantization(model, False)
        disable_quant_class.disable_param_quantization(model, False)
        return_quant_tensor_state = disable_return_quant_tensor(model)

    device = next(model.parameters()).device
    data_saver = DataSaverHookLLM(
        cache_args, cache_kwargs, cache_outs, store_args, store_kwargs, store_outs, keep_gpu)
    handle = module.register_forward_hook(data_saver, with_kwargs=True)
    with torch.no_grad():
        for inps in dataloader:
            try:
                inps = send_to_device(inps, device)
                model(**inps)
            except StopFwdException:
                pass
    handle.remove()
    if disable_quant:
        disable_quant_class.enable_act_quantization(model, False)
        disable_quant_class.enable_param_quantization(model, False)
        restore_return_quant_tensor(model, return_quant_tensor_state)


# TODO: Move imports to their appropiate place
import itertools

from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from brevitas.optim.sign_sgd import SignSGD


def get_blocks(model: nn.Module, block_check_fn: Callable[[nn.Module, str],
                                                          bool]) -> List[nn.Module]:
    blocks = []

    # Iterating over .modules() might have been more readable but
    # with this recursive implementation, once a block is reached,
    # its subtree of modules is not expanded.
    def _get_blocks(module: nn.Module):
        for module_name, module_child in module.named_children():
            if block_check_fn(module_child, module_name):
                blocks.append(module_child)
            else:
                _get_blocks(module_child)

    # Run recursive function that updates the list blocks
    _get_blocks(model)
    return blocks


def _is_block_llm(module: nn.Module, module_name: str) -> bool:
    return isinstance(module, LlamaDecoderLayer) or isinstance(module, OPTDecoderLayer)


def apply_learned_round_learning_llm(
    model: nn.Module,
    dataloader: DataLoader,
    learned_round: LearnedRound = AutoRound(iters=100),
    optimizer_class: Optimizer = SignSGD,
    iters: int = 100,
    optimizer_lr: float = 5e-3,
    block_check_fn: Callable = _is_block_llm,
):
    # Disable the cache to prevent memory buildup
    cache_state = model.config.use_cache
    model.config.use_cache = False
    # NOTE: Can be problematic is more than one GPU is used.
    device = next(model.parameters()).device
    # Retrieve blocks using the appropiate function to check blocks
    blocks = get_blocks(model, block_check_fn)

    print(f"Total Iterations per block {iters}")
    print(f"Number of blocks {len(blocks)}")

    cache_args, cache_kwargs, cache_outs = [], [], []

    for block_idx, (block, block_loss, block_learned_round_modules) in enumerate(
            learned_round.learned_round_iterator(blocks)):
        optimizer = optimizer = optimizer_class(
            itertools.chain(
                *[
                    learned_round_module.parameters()
                    for learned_round_module in block_learned_round_modules]),
            lr=optimizer_lr)
        # Cache needs to be cleaned between blocks. No need to clear the
        # kwargs cache, as this is only updates for the first block.
        cache_args = []
        cache_outs = []
        # Save FP output
        save_inp_out_data_llm(
            model,
            block,
            dataloader,
            cache_args,
            cache_kwargs,
            cache_outs,
            store_args=False,
            store_kwargs=False,
            store_outs=True,
            keep_gpu=True,
            disable_quant=True)
        # Save Quant input
        save_inp_out_data_llm(
            model,
            block,
            dataloader,
            cache_args,
            cache_kwargs,
            cache_outs,
            store_args=True,
            store_kwargs=len(cache_kwargs) == 0,
            store_outs=False,
            keep_gpu=True,
            disable_quant=False)

        pbar = tqdm(range(iters), desc='')
        for _ in pbar:
            idx = torch.randint(0, len(cache_args), (1,))
            args, kwargs, fp_out = cache_args[idx], cache_kwargs[idx], cache_outs[idx]
            block.train()

            optimizer.zero_grad()

            args = send_to_device(args, device)
            kwargs = send_to_device(kwargs, device)
            fp_out = send_to_device(fp_out, device)

            quant_out = block(*args, **kwargs)
            if isinstance(quant_out, tuple):
                quant_out = quant_out[0]

            loss, loss_components = block_loss(quant_out, fp_out)

            loss.backward()
            optimizer.step()
            # Update progress bar
            pbar.set_description(
                "block = {:d}/{:d}, {}".format(
                    block_idx + 1, len(blocks),
                    block_loss.format_loss_components(*loss_components)))
            pbar.update(1)

    # Restore cache state
    model.config.use_cache = cache_state
