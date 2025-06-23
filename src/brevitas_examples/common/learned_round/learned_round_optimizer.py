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

from abc import abstractmethod
from contextlib import nullcontext
import copy
from typing import (
    Any, Callable, Dict, Generic, Iterable, List, Optional, OrderedDict, Sequence, Tuple, TypeVar)
import warnings

from accelerate.utils.operations import send_to_device
import torch
from torch import autocast
from tqdm import tqdm

try:
    from torch import GradScaler
except:
    from torch.cuda.amp import GradScaler

from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from brevitas import config
from brevitas.core.function_wrapper.learned_round import LearnedRoundSte
from brevitas.graph.calibrate import quantization_status_manager
from brevitas.inject.enum import FloatToIntImplType
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.utils.torch_utils import StopFwdException
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.common.learned_round.learned_round_args import Config
from brevitas_examples.common.learned_round.learned_round_args import OptimizerArgs
from brevitas_examples.common.learned_round.learned_round_args import TARGET_PARAMETRIZATIONS_MAP
from brevitas_examples.common.learned_round.learned_round_method import BlockLoss
from brevitas_examples.common.learned_round.learned_round_method import LEARNED_ROUND_VALUE_INIT_MAP

# TODO: Remove
config.IGNORE_MISSING_KEYS = True

_T_inputs = TypeVar("_T_inputs")
_T_outputs = TypeVar("_T_output")
_T_cache = Tuple[_T_inputs, _T_outputs]


# Cache, as a subclass of torch.utils.data.Dataset, needs to implement __getitem__ and __len__
class Cache(Generic[_T_inputs, _T_outputs], Dataset[_T_cache]):

    inputs: Sequence[_T_inputs]
    outputs: Sequence[_T_outputs]

    @abstractmethod
    def store_inputs(self, args: Tuple[torch.Tensor, ...], kwargs: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def store_output(self, output: Any) -> None:
        pass

    @abstractmethod
    def reset_cache(self) -> None:
        pass

    @abstractmethod
    def collate_fn(self, batch: Iterable[_T_cache]) -> _T_cache:
        pass

    def collate_fn_output_next(self, batch: Iterable[_T_cache]) -> _T_cache:
        raise NotImplementedError(f"{self.__class__} is not compatible with fast_update=True.")

    def collate_fn_input_next(self, batch: Iterable[_T_cache]) -> _T_cache:
        raise NotImplementedError(f"{self.__class__} is not compatible with fast_update=True.")


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
        disable_quantization_cm = quantization_status_manager(
            model=model,
            disable_act_quant=True,
            disable_weight_quant=True,
            disable_bias_quant=True,
            is_training=False,
        )
    else:
        disable_quantization_cm = nullcontext()

    data_saver = DataSaverHook(
        cache, store_inputs=store_inputs, store_output=store_output, keep_gpu=keep_gpu)
    handle = module.register_forward_hook(data_saver, with_kwargs=True)
    with torch.no_grad(), disable_quantization_cm:
        for inps in dataloader:
            try:
                model_forward(model, inps)
            except StopFwdException:
                pass
    handle.remove()


class block_optimization_cm:
    """
        Context manager to prepare a block for optimization

    Args:
        model (nn.Module): module for which a subset of parameters are optimized
        target_params (nn.Parameter): subset of parameters to be optimized
    """

    def __init__(self, module: nn.Module, target_params: OrderedDict[str, nn.Parameter]) -> None:
        self.module = module
        self.target_params = target_params

    def __enter__(self) -> None:
        self.module.eval()
        # Move module to GPU if available
        if torch.cuda.is_available():
            try:
                self.module.cuda()
            except RuntimeError as exc:
                if 'out of memory' in str(exc):
                    warnings.warn(
                        "Out of memory error was raised when moving the model to GPU. Defaulting to CPU."
                    )
                else:
                    raise exc
        # Freeze parameters within the block that should not be optimized
        for name, param in self.module.named_parameters():
            param.requires_grad = name in self.target_params

    def __exit__(self, type, value, traceback) -> None:
        # After optimization, freeze all the parameters of the block
        for param in self.module.parameters():
            param.requires_grad = False
        # And module is moved back to CPU
        self.module.cpu()


class LearnedRoundOptimizer:

    def __init__(self, config: Config) -> None:
        self.config = config

    def _step(
            self,
            optim_lr_schedulers: List[Tuple[Optimizer, Optional[LRScheduler]]],
            scaler: Optional[GradScaler] = None) -> None:
        for optim, lr_scheduler in optim_lr_schedulers:
            if scaler is not None:
                scaler.step(optim)
            else:
                optim.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            optim.zero_grad()
        # Update ths scaler
        if scaler is not None:
            scaler.update()

    def _training_step(
        self,
        model: nn.Module,
        forward: Callable,
        loss_fn: BlockLoss,
        inputs: _T_cache,
        scaler: Optional[GradScaler] = None,
    ) -> Tuple[torch.Tensor, Any]:
        # Compute loss
        loss, loss_components = self._compute_loss(
            model,
            forward,
            loss_fn,
            inputs,
        )
        # Run backward, optionally scaling the loss
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss = loss * self.config.training_args.loss_scaling_factor
            loss.backward()

        return loss.detach().cpu().item(), loss_components

    def _amp_context_manager(self):
        if self.config.training_args.use_amp:
            ctx_manager = autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=self.config.training_args.amp_dtype)
        else:
            ctx_manager = nullcontext()

        return ctx_manager

    def _compute_loss(
        self,
        model: nn.Module,
        forward: Callable,
        loss_fn: BlockLoss,
        inputs: _T_cache,
    ) -> Tuple[torch.Tensor, Any]:
        # Unpack inputs to model
        inps, labels = inputs

        with self._amp_context_manager():
            # Run block forward to obtain quant outputs
            output = forward(model, inps)
            labels = send_to_device(labels, output.device)
            loss, loss_components = loss_fn(output, labels)

        return loss, loss_components

    def _get_target_parameters(
        self,
        model: nn.Module,
        get_target: Callable[[nn.Module, OrderedDict, str], bool],
        state_dict: Optional[OrderedDict[str,
                                         nn.Parameter]] = None) -> OrderedDict[str, nn.Parameter]:

        state_dict = OrderedDict() if state_dict is None else state_dict

        def _recursive_get_target_parameters(module: nn.Module, prefix: str = "") -> None:
            # Base case
            if get_target(module, state_dict, prefix):
                # Early stoppping
                return
            for child_name, child_module in module.named_children():
                _recursive_get_target_parameters(
                    child_module, f"{prefix}.{child_name}" if len(prefix) > 0 else f"{child_name}")

        # Run recursion from model
        _recursive_get_target_parameters(model)
        return state_dict

    def _get_target_params(self, model: nn.Module) -> OrderedDict:
        state_dict = OrderedDict()
        # Iterate over the target parameters
        get_target_param_fns = list(
            map(
                lambda target_param: TARGET_PARAMETRIZATIONS_MAP[target_param],
                self.config.training_args.optimizers_targets))
        for get_target_param_fn in get_target_param_fns:
            state_dict = self._get_target_parameters(model, get_target_param_fn, state_dict)
        return state_dict

    def _create_optimizer_and_scheduler(
        self,
        params: List[nn.Parameter],
        optimizer_args: OptimizerArgs,
    ) -> Tuple[Optimizer, Optional[LRScheduler]]:
        # Instantiate optimizer
        optimizer = optimizer_args.optimizer_cls(
            params=params, lr=optimizer_args.lr, **optimizer_args.optimizer_kwargs)
        # Instantiate learning rate schedu
        lr_scheduler_args = optimizer_args.lr_scheduler_args
        lr_scheduler = (
            lr_scheduler_args.lr_scheduler_cls(optimizer, **lr_scheduler_args.lr_scheduler_kwargs)
            if lr_scheduler_args is not None else None)
        return optimizer, lr_scheduler

    def _create_optimizers_lr_schedulers(
            self, model: nn.Module) -> List[Tuple[Optimizer, Optional[LRScheduler]]]:
        # Retrieve configuration for optimizers and target parameters
        optimizers_args, optimizer_targets = self.config.training_args.optimizers_args, self.config.training_args.optimizers_targets
        return list(
            map(
                lambda target_args: self._create_optimizer_and_scheduler(
                    self._get_target_parameters(model, TARGET_PARAMETRIZATIONS_MAP[target_args[0]]).
                    values(),
                    target_args[1]),
                zip(optimizer_targets, optimizers_args)))

    def _load_target_params(self, model: nn.Module, state_dict: OrderedDict) -> None:
        prev = config.REINIT_ON_STATE_DICT_LOAD
        config.REINIT_ON_STATE_DICT_LOAD = False
        _incompatible_keys = model.load_state_dict(state_dict=state_dict, strict=False)
        assert len(_incompatible_keys.unexpected_keys) == 0, f"The following unexpected keys were found when loading the model parameters: {_incompatible_keys.unexpected_keys}."
        config.REINIT_ON_STATE_DICT_LOAD = prev

    def _training_loop(
        self,
        model: nn.Module,
        forward: Callable,
        data_loader: DataLoader,
        loss_fn: BlockLoss,
    ) -> Tuple[float, int, int]:

        # Initialize optimizers and lr schedulers
        optim_lr_schedulers = self._create_optimizers_lr_schedulers(model)

        # Variables needed for printing
        best_loss = torch.finfo(torch.float).max
        init_loss = -1.0
        last_best_iter = self.config.training_args.iters

        scaler = None
        if self.config.training_args.use_amp:
            scaler = GradScaler()

        # Dictionary to store the rounding parameters yielding the lowest
        # training loss
        pbar = tqdm(range(self.config.training_args.iters), desc='')
        # Zero-grad before starting
        model.zero_grad()

        # Prepare iterator
        data_loader = iter(data_loader)

        for i in pbar:
            # Sample mini-batch from data loader
            inputs = next(data_loader)

            # Compute loss and gradients
            loss, loss_components = self._training_step(model, forward, loss_fn, inputs, scaler)

            # Save best parameters before taking gradient step
            curr_loss = loss
            init_loss = curr_loss if i == 0 else init_loss
            if loss < best_loss:
                best_loss = curr_loss
                last_best_iter = i + 1
                if self.config.training_args.use_best_model:
                    with torch.no_grad():
                        optimal_state_dict = copy.deepcopy(self._get_target_params(model))

            # Scale loss and perform gradient step
            self._step(optim_lr_schedulers, scaler)

            # Update progress bar
            pbar.set_description("{}".format(loss_fn.format_loss_components(*loss_components)))

        # Make sure no updates are received in the progress bar
        pbar.close()

        if self.config.training_args.use_best_model:
            self._load_target_params(model, optimal_state_dict)
        else:
            # Override if the model with the lowest training error is not used
            best_loss = curr_loss
            last_best_iter = self.config.training_args.iters

        return init_loss, best_loss, last_best_iter

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

    def _insert_learned_round_quantizers(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, QuantWBIOL) and len([
                    m for m in module.modules() if isinstance(m, LearnedRoundSte)]) == 0:
                value = LEARNED_ROUND_VALUE_INIT_MAP[
                    self.config.learned_round_args.learned_round_param.value](
                        module, **self.config.learned_round_args.learned_round_kwargs)
                module.weight_quant.quant_injector = module.weight_quant.quant_injector.let(
                    float_to_int_impl_type=FloatToIntImplType.LEARNED_ROUND,
                    learned_round_impl_type=self.config.learned_round_args.learned_round_param,
                    learned_round_init=value,
                    **self.config.learned_round_args.learned_round_kwargs,
                )
                module.weight_quant.init_tensor_quant(preserve_state_dict=True)

    def apply_learned_round(
            self,
            model: nn.Module,
            model_forward: Callable,
            block_forward: Callable,
            dataset: Dataset,
            cache: Cache,
            get_blocks_fn: Callable[[nn.Module], List[nn.Module]],
            collate_fn: Callable,
            model_prepare_fn: Optional[Callable] = None,
            model_finish_fn: Optional[Callable] = None,
            keep_gpu: bool = True) -> None:

        # Perform any needed preprocessing before rounding optimisation, e.g. disabling caching in LLMs
        model_dict = None if model_prepare_fn is None else model_prepare_fn(model)

        # Insert quantizers within the appropiate model blocks
        self._insert_learned_round_quantizers(model)

        # Retrieve blocks using the appropiate function to check blocks
        blocks = get_blocks_fn(model)

        print(f"Total Iterations per block {self.config.training_args.iters}")
        print(f"Number of blocks {len(blocks)}")

        # Iterate over blocks and optimise the rounding parameters within each of them
        for block_idx, block in enumerate(blocks):
            if block_idx == 0 or not self.config.learned_round_args.fast_update:
                # Cache needs to be cleared before populating it with the inputs and outputs
                # to the block under optimization.
                cache.reset_cache()
                # Distribute the model across devices to run a forward pass to capture
                # inputs/outputs to the given block
                model = offload_model(model)
                data_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
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

            # Loss function for computing the rounding loss within each block
            block_loss = self.config.learned_round_args.loss_cls(
                block, **self.config.learned_round_args.loss_kwargs)

            # Per-block training data loader
            block_data_loader = DataLoader(
                cache,
                batch_size=self.config.training_args.batch_size,
                sampler=torch.utils.data.RandomSampler(
                    data_source=cache,
                    replacement=True,
                    num_samples=self.config.training_args.batch_size *
                    self.config.training_args.iters),
                collate_fn=cache.collate_fn)

            # Optimize block
            with block_optimization_cm(module=block, target_params=self._get_target_params(block)):
                init_loss, best_loss, last_best_iter = self._training_loop(
                    model=block,
                    forward=block_forward,
                    data_loader=block_data_loader,
                    loss_fn=block_loss,
                )

            print(
                f"Quantized block {block_idx+1}/{len(blocks)}, "
                f"initial loss: {init_loss:.6f}, best loss: {best_loss:.6f}, at iteration {last_best_iter}."
            )

            if block_idx + 1 < len(blocks) and self.config.learned_round_args.fast_update:
                cache = self.skip_full_execution(block, blocks[block_idx + 1], block_forward, cache)

        # The original configuration of the model is restored after finishing the optimization
        if model_finish_fn is not None:
            model_finish_fn(model, model_dict)

    def skip_full_execution(
            self, block: nn.Module, next_block: nn.Module, block_forward: Callable, cache: Cache):
        # We need to compute two inputs, one is a floating point one to compute float out
        # The second is a quantized one to create the quantized input of the next blocks

        # Temporary caches
        cache_fp_outputs = type(cache)()
        cache_quant_inputs = type(cache)()

        # Prepare floating point inputs for next block
        output_next_data_loader = DataLoader(
            cache, batch_size=1, collate_fn=cache_fp_outputs.collate_fn_output_next)

        # We compute the floating point output of the upcoming block
        if torch.cuda.is_available():
            next_block.cuda()

        # Save floating point output of next block in cache_fp_outputs
        save_inputs_output(
            next_block,
            block_forward,
            next_block,
            output_next_data_loader,
            cache_fp_outputs,
            store_inputs=False,
            store_output=True,
            keep_gpu=False,
            disable_quant=True,
        )
        next_block.cpu()

        # Prepare quant inputs to current block
        input_next_data_loader = DataLoader(
            cache, batch_size=1, collate_fn=cache_quant_inputs.collate_fn_input_next)

        # Finally (!), we compute the quantized input of the next block
        block.eval()
        if torch.cuda.is_available():
            block.cuda()

        save_inputs_output(
            block,
            block_forward,
            block,
            input_next_data_loader,
            cache_quant_inputs,
            store_inputs=False,
            store_output=True,
            keep_gpu=False,
            disable_quant=False,
        )

        block.cpu()

        # Update the cache with the (inputs, outputs) in the temporary caches
        cache.inputs = cache_quant_inputs.outputs
        cache.outputs = cache_fp_outputs.outputs

        return cache
