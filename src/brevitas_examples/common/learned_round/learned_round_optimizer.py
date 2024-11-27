"""
Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

Adapted from https://github.com/intel/auto-round, released under the following LICENSE:

                              Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS
"""

from abc import ABC
from abc import abstractmethod
import copy
from functools import partial
import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import warnings

from accelerate import Accelerator
from accelerate.utils import tqdm as tqdm_accelerate
from accelerate.utils.dataclasses import PrecisionType
from accelerate.utils.operations import send_to_device
from datasets import Dataset
import torch
from torch import autocast
from torch import nn
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.sgd import SGD
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from brevitas import config
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.graph.calibrate import disable_return_quant_tensor
from brevitas.graph.calibrate import DisableEnableQuantization
from brevitas.graph.calibrate import restore_return_quant_tensor
from brevitas.optim.sign_sgd import SignSGD
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjectorBase
from brevitas.utils.torch_utils import StopFwdException
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.common.learned_round.learned_round_method import LearnedRound
from brevitas_examples.common.learned_round.learned_round_method import LearnedRoundLoss

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


def return_scale_parameters(block: nn.Module) -> List[nn.Parameter]:

    scale_parameters = []

    def _get_scale_parameters(module: nn.Module):
        for module_child in module.children():
            if isinstance(module, WeightQuantProxyFromInjectorBase):
                for submodule_name, submodule in module_child.named_parameters():
                    if submodule_name.endswith('scaling_impl.value'):
                        scale_parameters.append(submodule)
            else:
                _get_scale_parameters(module_child)

    # Run recursion from block
    _get_scale_parameters(block)
    return scale_parameters


class Cache(ABC):

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def store_inputs(self, args: Any, kwargs: Any) -> None:
        pass

    @abstractmethod
    def store_output(self, output: Any) -> None:
        pass

    @abstractmethod
    def sample_batch(self, indices: torch.Tensor) -> Union[Any, torch.Tensor]:
        pass

    @abstractmethod
    def initialize_cache(self) -> None:
        pass

    @abstractmethod
    def clear_cache(self) -> None:
        pass

    @abstractmethod
    def cache_to_dataset(self) -> Dataset:
        pass

    @abstractmethod
    def collate_fn(self, batch: Any) -> Any:
        pass


class DataSaverHook:

    def __init__(
        self,
        cache: Cache,
        store_inputs: bool = True,
        store_output: bool = True,
        keep_gpu: bool = True,
    ) -> None:
        self.cache = cache
        self.store_inputs = store_inputs
        self.store_output = store_output
        self.keep_gpu = keep_gpu

    def __call__(self, module, args, kwargs, output) -> None:
        if self.store_inputs:
            if not self.keep_gpu:
                args = send_to_device(args, 'cpu')
                kwargs = send_to_device(kwargs, 'cpu')
            self.cache.store_inputs(args, kwargs)
        if self.store_output:
            if not self.keep_gpu:
                output = send_to_device(output, 'cpu')
            self.cache.store_output(output)

        raise StopFwdException


def save_inputs_output(
        model: nn.Module,
        model_forward: Callable,
        module: nn.Module,
        dataloader: DataLoader,
        cache: Cache,
        store_inputs: bool = True,
        store_output: bool = False,
        keep_gpu: bool = True,
        disable_quant: bool = False) -> None:
    if disable_quant:
        disable_quant_class = DisableEnableQuantization()
        disable_quant_class.disable_act_quantization(model, False)
        disable_quant_class.disable_param_quantization(model, False)
        return_quant_tensor_state = disable_return_quant_tensor(model)

    data_saver = DataSaverHook(
        cache, store_inputs=store_inputs, store_output=store_output, keep_gpu=keep_gpu)
    handle = module.register_forward_hook(data_saver, with_kwargs=True)
    with torch.no_grad():
        for inps in dataloader:
            try:
                model_forward(model, inps)
            except StopFwdException:
                pass
    handle.remove()
    if disable_quant:
        disable_quant_class.enable_act_quantization(model, False)
        disable_quant_class.enable_param_quantization(model, False)
        restore_return_quant_tensor(model, return_quant_tensor_state)


class LearnedRoundOptimizer:

    def __init__(
        self,
        learned_round: LearnedRound,
        learned_round_loss_class: Type[LearnedRoundLoss],
        *,
        optimizer_class: Type[Optimizer] = SignSGD,
        scale_optimizer_class: Type[Optimizer] = SGD,
        lr_scheduler_class: Optional[Type[LRScheduler]] = LinearLR,
        optimizer_lr: float = 5e-3,
        batch_size: float = 8,
        iters: int = 200,
        learn_scale: bool = False,
        use_best_model: bool = True,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        loss_scaling_factor: float = 1000.,
        use_accelerate: bool = False,
        learned_round_loss_kwargs: Optional[Dict] = None,
        optimizer_kwargs: Optional[Dict] = None,
        lr_scheduler_kwargs: Optional[Dict] = None,
    ) -> None:
        self.learned_round = learned_round
        self.optimizer_class = optimizer_class
        self.scale_optimizer_class = scale_optimizer_class
        self.lr_scheduler_class = lr_scheduler_class
        self.optimizer_lr = optimizer_lr
        self.batch_size = batch_size
        self.iters = iters
        self.learn_scale = learn_scale
        self.use_best_model = use_best_model
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.loss_scaling_factor = loss_scaling_factor
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.lr_scheduler_kwargs = {} if lr_scheduler_kwargs is None else lr_scheduler_kwargs
        self.lr_scheduler_kwargs["total_iters"] = self.iters

        learned_round_loss_kwargs = {} if learned_round_loss_kwargs is None else learned_round_loss_kwargs
        self.learned_round_loss_init = partial(
            learned_round_loss_class, **learned_round_loss_kwargs)

        # TODO: Remove once validated and expose the flag
        # self.use_accelerate = use_accelerate
        self.use_accelerate = False

    @torch.no_grad()
    def _load_round_params(self, block: nn.Module, round_params: Dict) -> None:
        for n, m in block.named_modules():
            if n in round_params:
                m.load_state_dict(round_params[n])

    @torch.no_grad()
    def _collect_round_params(self, block: nn.Module) -> Dict:
        params = {}
        for n, m in block.named_modules():
            if isinstance(m, LearnedRoundSte):
                params[n] = copy.deepcopy(m.state_dict())
        return params

    def _optim_step(self, *optimizers: Optimizer) -> None:
        for optimizer in optimizers:
            if optimizer:
                optimizer.step()
                optimizer.zero_grad()

    def _lr_sched_step(self, *lr_schedulers: LRScheduler) -> None:
        for lr_scheduler in lr_schedulers:
            if lr_scheduler:
                lr_scheduler.step()

    def _step(self, optimizers: List[Optimizer], lr_schedulers: List[LRScheduler]) -> None:
        for optimizer in optimizers:
            if optimizer:
                optimizer.step()
                optimizer.zero_grad()
        for lr_scheduler in lr_schedulers:
            if lr_scheduler:
                lr_scheduler.step()

    def _populate_cache(
        self,
        cache: Cache,
        model: nn.Module,
        model_forward: nn.Module,
        block: nn.Module,
        data_loader: DataLoader,
        keep_gpu: bool = True,
        capture_quant_input: bool = True,
        capture_quant_output: bool = False,
    ) -> None:
        # Populate the cache with new inputs and outputs
        save_inputs_output(
            model,
            model_forward,
            block,
            data_loader,
            cache,
            store_inputs=True,
            store_output=capture_quant_input == capture_quant_output,
            keep_gpu=keep_gpu,
            disable_quant=not capture_quant_input,
        )
        if capture_quant_input != capture_quant_output:
            save_inputs_output(
                model,
                model_forward,
                block,
                data_loader,
                cache,
                store_inputs=False,
                store_output=True,
                keep_gpu=keep_gpu,
                disable_quant=not capture_quant_output,
            )

    def _optimize_learned_round_block(
        self,
        block: nn.Module,
        block_learned_round_modules: List[nn.Module],
        cache: Cache,
        block_loss: LearnedRoundLoss,
        block_forward: Callable,
        scale_params: Optional[nn.Parameter] = None,
    ) -> Tuple[float, float, int]:
        # Move block to GPU if available
        if torch.cuda.is_available():
            try:
                block.cuda()
            except RuntimeError as exc:
                if 'out of memory' in str(exc):
                    warnings.warn(
                        "Out of memory error was raised when moving the block to GPU. Defaulting to CPU."
                    )
                else:
                    raise exc

        # Initialize optimizer and LR scheduler
        optimizer = self.optimizer_class(
            itertools.chain(
                *[
                    block_learned_round_module.parameters()
                    for block_learned_round_module in block_learned_round_modules]),
            lr=self.optimizer_lr,
            **self.optimizer_kwargs,
        )
        lr_scheduler = (
            self.lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
            if self.lr_scheduler_class else None)

        # Initialize optimizer/LR scheduler for the scale parameters if enabled
        if self.learn_scale and scale_params is not None:
            optimizer_scale = self.scale_optimizer_class(
                scale_params,
                lr=self.optimizer_lr,
                momentum=0.9,
                **self.optimizer_kwargs,
            )
            lr_scheduler_scale = (
                self.lr_scheduler_class(
                    optimizer_scale, start_factor=1, end_factor=0, total_iters=600)
                if self.lr_scheduler_class else None)
        else:
            optimizer_scale = None
            lr_scheduler_scale = None

        # Variables needed for printing
        best_loss = torch.finfo(torch.float).max
        init_loss = -1.0
        last_best_iter = self.iters

        # Dictionary to store the rounding parameters yielding the lowest
        # training loss
        optimal_rounding_params = {}
        torch.autograd.set_detect_anomaly(True)
        n_samples = len(cache)
        pbar = tqdm(range(self.iters), desc='')
        for i in pbar:
            # Sample mini-batch from cache
            idxs = torch.randperm(n_samples)[:self.batch_size]
            inputs, fp_outs = cache.sample_batch(idxs)

            # Run block forward to obtain quant outputs
            quant_outs = block_forward(block, inputs)
            fp_outs = send_to_device(fp_outs, quant_outs.device)
            if self.use_amp:
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                              dtype=self.amp_dtype):
                    loss, loss_components = block_loss(quant_outs, fp_outs)
            else:
                loss, loss_components = block_loss(quant_outs.to(torch.float32), fp_outs.to(torch.float32))

            # Save best parameters before taking gradient step
            curr_loss = loss.detach().cpu().item()
            init_loss = curr_loss if i == 0 else init_loss
            if loss < best_loss:
                best_loss = curr_loss
                last_best_iter = i + 1
                if self.use_best_model:
                    optimal_rounding_params = self._collect_round_params(block)

            # Scale loss and perform gradient step
            loss = loss * self.loss_scaling_factor
            loss.backward()
            self._optim_step(optimizer, optimizer_scale)
            self._lr_sched_step(lr_scheduler, lr_scheduler_scale)

            # Update progress bar
            pbar.set_description("{}".format(block_loss.format_loss_components(*loss_components)))

        # Make sure no updates are received in the progress bar
        pbar.close()

        if self.use_best_model:
            self._load_round_params(block, optimal_rounding_params)
        else:
            # Override if the model with the lowest training error is not used
            best_loss = curr_loss
            last_best_iter = self.iters

        # Move the block back to CPU
        block.cpu()

        return init_loss, best_loss, last_best_iter

    # TODO: Enable saving best parameters
    def _accelerate_optimize_learned_round_block(
        self,
        block: nn.Module,
        block_learned_round_modules: List[nn.Module],
        cache: Cache,
        block_loss: LearnedRoundLoss,
        block_forward: Callable,
    ) -> Tuple[float, float, int]:
        # Enable running in mixed precision
        TORCH_DTYPE_TO_PRECISION_TYPE_MAP = {
            torch.float16: PrecisionType.FP16,
            torch.bfloat16: PrecisionType.BF16,}
        raise_warning_dtype = False
        if not self.use_amp:
            mixed_precision_type = None
        else:
            if self.amp_dtype not in TORCH_DTYPE_TO_PRECISION_TYPE_MAP:
                raise_warning_dtype = True
                mixed_precision_type = None
            else:
                mixed_precision_type = TORCH_DTYPE_TO_PRECISION_TYPE_MAP[self.amp_dtype]
        # Instantiate accelerator to run in a multi-GPU setting
        accelerator = Accelerator(mixed_precision=mixed_precision_type)

        # Raise warning if the AMP dtype was defaulted to float32. This warning is raised after
        # the instantiation of accelerator, to use its print functionality so the message is only
        # printed once.
        if raise_warning_dtype:
            accelerator.print(
                f"The dtype {self.amp_dtype} cannot be used for AMP training with accelerate. Defaulting to float32."
            )

        # Initilalize optimizer and LR scheduler
        optimizer = self.optimizer_class(
            itertools.chain(
                *[
                    block_learned_round_module.parameters()
                    for block_learned_round_module in block_learned_round_modules]),
            lr=self.optimizer_lr,
            **self.optimizer_kwargs,
        )
        lr_scheduler = (
            self.lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
            if self.lr_scheduler_class else None)

        # Prepare dataset from cache
        cache_dataset = cache.cache_to_dataset()
        cache_dataloader = DataLoader(
            cache_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=cache.collate_fn)

        # Prepare elements for training
        cache_dataloader, block, optimizer, lr_scheduler = accelerator.prepare(cache_dataloader, block, optimizer, lr_scheduler)

        # Variables needed for printing
        best_loss = torch.finfo(torch.float).max
        init_loss = -1.0
        last_best_iter = self.iters

        # Initialize an iterator to extract elements from the cache dataloader
        cache_iterator = iter(cache_dataloader)

        pbar = tqdm_accelerate(range(self.iters), desc='')
        for i in pbar:
            # Sample mini-batch from cache
            inputs, fp_outs = next(cache_iterator)

            # Run block forward to obtain quant outputs
            quant_outs = block_forward(block, inputs)
            # Compute loss using the block loss function
            loss, loss_components = block_loss(quant_outs, fp_outs)

            # Save best parameters before taking gradient step
            curr_loss = loss.detach().cpu().item()
            init_loss = curr_loss if i == 0 else init_loss
            if loss < best_loss:
                best_loss = curr_loss
                last_best_iter = i + 1

            # Scale loss and perform gradient step
            # loss = loss * self.loss_scaling_factor
            accelerator.backward(loss)
            self._step(optimizer, lr_scheduler)

            # Update progress bar
            pbar.set_description("{}".format(block_loss.format_loss_components(*loss_components)))

        # Make sure no updates are received in the progress bar
        pbar.close()

        # TODO: Include support for saving the best configuration during training
        if not self.use_best_model:
            # Override if the model with the lowest training error is not used
            best_loss = curr_loss
            last_best_iter = self.iters

        # TODO: Verify if this call is actually needed
        # Wait for everyone before proceding to next block
        accelerator.wait_for_everyone()
        # Remove all the wrapper around the block
        block = accelerator.unwrap_model(block)
        # Clear memory
        accelerator.free_memory()
        # Move the block back to CPU
        block.cpu()

        return init_loss, best_loss, last_best_iter

    def apply_learned_round(
            self,
            model: nn.Module,
            model_forward: Callable,
            block_forward: Callable,
            data_loader: DataLoader,
            cache: Cache,
            get_blocks_fn: Callable,
            model_prepare_fn: Optional[Callable] = None,
            model_finish_fn: Optional[Callable] = None,
            keep_gpu: bool = True) -> None:

        # Perform any needed preprocessing before rounding optimisation, e.g. disabling caching in LLMs
        model_dict = None if model_prepare_fn is None else model_prepare_fn(model)

        # Insert quantizers within the appropiate model blocks
        self.learned_round.insert_learned_round_quantizers(model)

        # Retrieve blocks using the appropiate function to check blocks
        blocks = get_blocks_fn(model)

        print(f"Total Iterations per block {self.iters}")
        print(f"Number of blocks {len(blocks)}")

        # Initialize cache to store partial inputs and outputs for each block
        cache.initialize_cache()

        # Iterate over blocks and optimise the rounding parameters within each of them
        for block_idx, block in enumerate(blocks):
            # Distribute the model across devices to run a forward pass to capture
            # inputs/outputs to the given block
            model = offload_model(model)
            # Cache needs to be cleared before populating it with the inputs and outputs
            # to the block under optimization.
            self._populate_cache(
                cache,
                model,
                model_forward,
                block,
                data_loader,
                keep_gpu=keep_gpu,
                capture_quant_input=True,
                capture_quant_output=False,
            )
            # Remove hooks needed to offload the model blocks to cpu
            remove_hooks(model)

            # Retrieve scales
            scale_params = return_scale_parameters(block)

            # The parameters of the block that are not part of the rounding quantizers
            # need to be frozen, as only the rounding needs to be optimized.
            block.eval()
            for params in block.parameters():
                params.requires_grad = False
            # However, the rounding parameters are tuned
            block_learned_round_modules = self.learned_round.return_learned_round_quantizers(block)
            for block_learned_round_module in block_learned_round_modules:
                block_learned_round_module.train()
                for params in block_learned_round_module.parameters():
                    params.requires_grad = True
            # As well as the scale parameters, if enabled
            if self.learn_scale:
                for params in scale_params:
                    params.requires_grad = True

            # Move block to GPU if available
            if torch.cuda.is_available():
                block.cuda()

            # Loss function for computing the rounding loss within each block
            block_loss = self.learned_round_loss_init(
                block,
                block_learned_round_modules,
            )

            # Optimize block rounding
            init_loss, best_loss, last_best_iter = (
                self._optimize_learned_round_block
                if not self.use_accelerate
                else self._accelerate_optimize_learned_round_block
            )(
                block=block,
                block_learned_round_modules=block_learned_round_modules,
                cache=cache,
                block_loss=block_loss,
                block_forward=block_forward,
                scale_params=scale_params,
            )

            print(
                f"Quantized block {block_idx+1}/{len(blocks)}, "
                f"initial loss: {init_loss:.6f}, best loss: {best_loss:.6f}, at iteration {last_best_iter}."
            )

            # After finishing the optimization, the block rounding parameters are frozen
            for block_learned_round_module in block_learned_round_modules:
                block_learned_round_module.eval()
                for params in block_learned_round_module.parameters():
                    params.requires_grad = False
            for params in scale_params:
                params.requires_grad = False

            # Move the block back to CPU
            block.cpu()

            # TODO: This call might not be needed, check_clear and reset_cache methods
            # Reset cache after optimisation
            cache.clear_cache()

        # The original configuration of the model is restored after finishing the optimization
        if model_finish_fn is not None:
            model_finish_fn(model, model_dict)
