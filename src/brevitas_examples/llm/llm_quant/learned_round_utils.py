# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, List, Tuple

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTDecoderLayer

from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.graph.calibrate import restore_return_quant_tensor
from brevitas_examples.common.learned_round.learned_round_method import StopFwdException
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundModelUtils


class LearnedRoundLLMUtils(LearnedRoundModelUtils):

    def __init__(self, loss_scaling_factor: float = 1000.) -> None:
        super(LearnedRoundLLMUtils, self).__init__()
        self.llm_cache_state = None
        self.loss_scaling_factor = loss_scaling_factor

    def default_block_check_fn(self, module: nn.Module, module_name: str) -> bool:
        return isinstance(module, LlamaDecoderLayer) or isinstance(module, OPTDecoderLayer)

    class _DataSaverHookLLM:

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

    def _save_inp_out_data(
            self,
            model: nn.Module,
            module: nn.Module,
            dataloader: DataLoader,
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

        device = next(module.parameters()).device
        data_saver = LearnedRoundLLMUtils._DataSaverHookLLM(
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

    def init_model_learned_round(self, model: nn.Module) -> None:
        self.llm_cache_state = model.config.use_cache
        model.config.use_cache = False

    def finish_model_learned_round(self, model: nn.Module) -> None:
        model.config.use_cache = self.llm_cache_state
        self.llm_cache_state = None

    def init_cache(self) -> Any:
        # cache_args, cache_kwargs, cache_outs
        return [], [], []

    def populate_cache(
        self,
        cache: Any,
        model: nn.Module,
        block: nn.Module,
        data_loader: DataLoader,
        keep_gpu: bool = True,
        **kwargs,
    ) -> int:
        # Unpack cache
        cache_args, cache_kwargs, cache_outs = cache
        # Cache needs to be cleaned between blocks. No need to clear the
        # kwargs cache, as this is only updated for the first block.
        cache_args.clear()
        cache_outs.clear()
        # Save FP output
        self._save_inp_out_data(
            model,
            block,
            data_loader,
            cache_args,
            cache_kwargs,
            cache_outs,
            store_args=False,
            store_kwargs=False,
            store_outs=True,
            keep_gpu=keep_gpu,
            disable_quant=True)
        # Save Quant input
        self._save_inp_out_data(
            model,
            block,
            data_loader,
            cache_args,
            cache_kwargs,
            cache_outs,
            store_args=True,
            store_kwargs=len(cache_kwargs) == 0,
            store_outs=False,
            keep_gpu=keep_gpu,
            disable_quant=False)
        # Return number of samples in calibration set
        return len(cache_args)

    def sample_cache(
        self,
        block: nn.Module,
        cache: Any,
        indices: torch.Tensor,
        input_dim: int = 0,
        **kwargs_fn,
    ) -> Tuple[Any, torch.Tensor]:
        cache_args, cache_kwargs, cache_outs = cache
        device = next(block.parameters()).device
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
                kwargs[key] = torch.cat(kwargs[key], dim=input_dim)
        # FP outputs
        outs = torch.cat([cache_outs[i] for i in indices], dim=input_dim)
        # Make sure that the inputs and outputs are in the same device as block,
        # before running its forward pass.
        args = send_to_device(args, device)
        kwargs = send_to_device(kwargs, device)
        outs = send_to_device(outs, device)

        return (args, kwargs), outs

    def run_forward(
        self,
        block: nn.Module,
        inputs: Any,
    ) -> torch.Tensor:
        args, kwargs = inputs
        quant_outs = block(*args, **kwargs)
        if isinstance(quant_outs, tuple):
            quant_outs = quant_outs[0]
        return quant_outs

    def loss_scaler(
        self,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        return loss * self.loss_scaling_factor
