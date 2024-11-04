# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
import copy
import itertools
from typing import Any, Callable, Dict, List, Tuple
import warnings

import torch
from torch import autocast
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from brevitas import config
from brevitas.optim.sign_sgd import SignSGD
from brevitas_examples.common.learned_round.learned_round_method import LearnedRound

config.IGNORE_MISSING_KEYS = True

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

class LearnedRoundModelUtils(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def default_block_check_fn(self, module: nn.Module, module_name: str) -> bool:
        pass

    @abstractmethod
    def init_model_learned_round(self, model: nn.Module) -> None:
        pass

    @abstractmethod
    def finish_model_learned_round(self, model: nn.Module) -> None:
        pass

    @abstractmethod
    def init_cache(self) -> Any:
        pass

    @abstractmethod
    def populate_cache(
        self,
        cache: Any,
        model: nn.Module,
        block: nn.Module,
        data_loader: DataLoader,
        keep_gpu: bool = True,
        **kwargs,
    ) -> int:
        pass

    @abstractmethod
    def sample_cache(
        self,
        block: nn.Module,
        cache: Any,
        indices: torch.Tensor,
        **kwargs,
    ) -> Tuple[Any, torch.Tensor]:
        pass

    @abstractmethod
    def run_forward(
        self,
        block: nn.Module,
        inputs: Any,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def loss_scaler(
        self,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        pass

class LearnedRoundOptimizer:

    def __init__(
        self,
        learned_round: LearnedRound,
        learned_round_utils: LearnedRoundModelUtils,
        optimizer_class: Optimizer = SignSGD,
        lr_scheduler_class: LRScheduler = LinearLR,
        optimizer_lr: float = 5e-3,
        batch_size: float = 8,
        iters: int = 200,
        use_best_model: bool = True,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        optimizer_kwargs: Dict = {},
        lr_scheduler_kwargs : Dict = {
            "start_factor": 1.0,
            "end_factor": 0.0,
            "verbose": False,
        }
    ) -> None:
        if learned_round.iters != iters:
            warnings.warn(
                "The number of iterations passed to the learned round optimiser is different "
                "to that of the learned round method, which might lead to unexpected behaviour."
            )
        self.learned_round = learned_round
        self.learned_round_utils = learned_round_utils
        self.optimizer_class = optimizer_class
        self.lr_scheduler_class = lr_scheduler_class
        self.optimizer_lr = optimizer_lr
        self.batch_size = batch_size
        self.iters = iters
        self.use_best_model = use_best_model
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.optimizer_kwargs = optimizer_kwargs

        self.lr_scheduler_kwargs = lr_scheduler_kwargs
        self.lr_scheduler_kwargs["total_iters"] = self.iters

    @torch.no_grad()
    def _load_round_params(self, block: nn.Module, round_params: Dict) -> None:
        for n, m in block.named_modules():
            if n in round_params:
                m.load_state_dict(round_params[n])

    @torch.no_grad()
    def _collect_round_params(self, block: nn.Module) -> Dict:
        params = {}
        for n, m in block.named_modules():
            if self.learned_round._is_learned_round_module(m):
                params[n] = copy.deepcopy(m.state_dict())
        return params

    def _scale_loss_and_backward(self, loss: torch.Tensor) -> torch.Tensor:
        scaled_loss = self.learned_round_utils.loss_scaler(loss)
        scaled_loss.backward()
        return scaled_loss

    def _step(self, optimizer: Optimizer, lr_scheduler: LRScheduler) -> None:
        optimizer.step()
        optimizer.zero_grad()
        if lr_scheduler:
            lr_scheduler.step()

    def apply_learned_round(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        block_check_fn: Callable = None,
        keep_gpu: bool = True
    ) -> None:
        # Prepare model for optimization
        self.learned_round_utils.init_model_learned_round(model)

        block_check_fn = block_check_fn if block_check_fn else self.learned_round_utils.default_block_check_fn
        # Retrieve blocks using the appropiate function to check blocks
        blocks = get_blocks(model, block_check_fn)

        print(f"Total Iterations per block {self.iters}")
        print(f"Number of blocks {len(blocks)}")

        # Initialise cache to store partial inputs and outputs for each block
        cache = self.learned_round_utils.init_cache()

        # Loop across blocks to optimise rounding within each
        for block_idx, (block, block_loss, block_learned_round_modules) in enumerate(
            self.learned_round.learned_round_iterator(blocks)):
            # Block needs to be in eval mode while the rounding is optimised
            block.eval()

            # Initialise optimiser and LR scheduler
            optimizer = self.optimizer_class(
                itertools.chain(
                    *[
                        learned_round_module.parameters()
                        for learned_round_module in block_learned_round_modules
                    ]
                ),
                lr=self.optimizer_lr,
                **self.optimizer_kwargs,
            )
            lr_scheduler = (
                self.lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
                if self.lr_scheduler_class
                else None
            )

            # Variables needed for printing
            best_loss = torch.finfo(torch.float).max
            init_loss = -1.0
            last_best_iter = self.iters

            optimal_rounding_params = {}

            torch.cuda.empty_cache()
            # Populate cache for the given block
            n_samples = self.learned_round_utils.populate_cache(
                cache,
                model,
                block,
                data_loader,
                keep_gpu=keep_gpu,
            )

            pbar = tqdm(range(self.iters), desc='')
            for i in pbar:
                # Sample mini-batch from cache
                idxs = torch.randperm(n_samples)[:self.batch_size]
                inputs, fp_outs = self.learned_round_utils.sample_cache(block, cache, idxs)

                # Run block forward to obtain quant outputs
                quant_outs = self.learned_round_utils.run_forward(block, inputs)

                if self.use_amp:
                    with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=self.amp_dtype):
                        loss, loss_components = block_loss(quant_outs, fp_outs)
                else:
                    loss, loss_components = block_loss(quant_outs.to(torch.float32), fp_outs.to(torch.float32))

                init_loss = loss.item() if i == 0 else init_loss

                if loss < best_loss:
                    best_loss = loss.item()
                    last_best_iter = i + 1
                    if self.use_best_model:
                        optimal_rounding_params = self._collect_round_params(block)

                # Scale loss and perform gradient step
                self._scale_loss_and_backward(loss)
                self._step(optimizer, lr_scheduler)

                 # Update progress bar
                pbar.set_description(
                    "Block = {:d}/{:d}, {}".format(
                        block_idx + 1, len(blocks),
                        block_loss.format_loss_components(*loss_components)))
                pbar.update(1)

            if self.use_best_model:
                self._load_round_params(block, optimal_rounding_params)
            else:
                # Override if the model with the lowest training error is not used
                best_loss = loss.item()
                last_best_iter = self.iters

            print(
                f"Quantized block {block_idx+1}/{len(blocks)}, "
                f"loss iter 0: {init_loss:.6f} -> iter {last_best_iter}: {best_loss:.6f}"
            )

        # Finish optimisation
        self.learned_round_utils.finish_model_learned_round(model)
