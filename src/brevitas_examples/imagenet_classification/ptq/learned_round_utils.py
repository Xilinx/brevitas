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
from typing import Any, Tuple

from accelerate.utils.operations import send_to_device
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from brevitas import config
from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.graph.calibrate import restore_return_quant_tensor
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.quant_tensor import QuantTensor
from brevitas_examples.common.learned_round.learned_round_method import StopFwdException
from brevitas_examples.common.learned_round.learned_round_optimizer import LearnedRoundModelUtils

config.IGNORE_MISSING_KEYS = True


class LearnedRoundVisionUtils(LearnedRoundModelUtils):

    def __init__(self) -> None:
        pass

    def init_model_learned_round(self, model: nn.Module) -> None:
        pass

    def finish_model_learned_round(self, model: nn.Module) -> None:
        pass

    def default_block_check_fn(self, module: nn.Module, module_name: str) -> bool:
        return (re.search(r"layer\d+", module_name) is not None)

    class _DataSaverHook:

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

    def _save_inp_out_data(
            self,
            model: nn.Module,
            module: nn.Module,
            dataloader: DataLoader,
            store_inp: bool = False,
            store_out: bool = False,
            keep_gpu: bool = True,
            disable_quant: bool = False):
        if disable_quant:
            disable_quant_class = DisableEnableQuantization()
            disable_quant_class.disable_act_quantization(model, False)
            disable_quant_class.disable_param_quantization(model, False)
            return_quant_tensor_state = disable_return_quant_tensor(model)

        device = next(model.parameters()).device
        data_saver = LearnedRoundVisionUtils._DataSaverHook(store_output=store_out)
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

    def init_cache(self) -> Any:
        return [], []

    def populate_cache(
        self,
        cache: Any,
        model: nn.Module,
        block: nn.Module,
        data_loader: DataLoader,
        keep_gpu: bool = True,
        **kwargs,
    ) -> int:
        cache_input, cache_output = cache
        # Clear caches
        cache_input.clear()
        cache_output.clear()

        _, all_fp_out = self._save_inp_out_data(model, block, data_loader, store_inp=False, store_out=True, keep_gpu=keep_gpu, disable_quant=True)
        all_quant_inp, _ = self._save_inp_out_data(model, block, data_loader, store_inp=True, store_out=True, keep_gpu=keep_gpu, disable_quant=False)

        # Add elements to the caches
        cache_input.append(all_quant_inp)
        cache_output.append(all_fp_out)

        # Number of samples
        return all_fp_out.shape[0]

    def sample_cache(
        self,
        block: nn.Module,
        cache: Any,
        indices: torch.Tensor,
        **kwargs,
    ) -> Tuple[Any, torch.Tensor]:
        cache_input, cache_output = cache
        device = next(block.parameters()).device

        input, output = cache_input[0][indices], cache_output[0][indices]
        input = send_to_device(input, device)
        output = send_to_device(output, device)

        return input, output

    def run_forward(
        self,
        block: nn.Module,
        inputs: Any,
    ) -> torch.Tensor:
        return block(inputs)

    def loss_scaler(
        self,
        loss: torch.Tensor,
    ) -> torch.Tensor:
        return loss
