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
import itertools
from typing import Any, Callable, Dict, List, Tuple
import warnings

from brevitas_examples.common.accelerate_utils.accelerate import offload_model, remove_hooks
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
from brevitas.proxy import WeightQuantProxyFromInjectorBase
from brevitas.utils.python_utils import recurse_getattr
from brevitas_examples.common.learned_round.learned_round_method import LearnedRound

config.IGNORE_MISSING_KEYS = True


def get_blocks(model: nn.Module, block_name: Callable[[nn.Module, str], bool]) -> List[nn.Module]:
    # blocks = []

    # # Iterating over .modules() might have been more readable but
    # # with this recursive implementation, once a block is reached,
    # # its subtree of modules is not expanded.
    # def _get_blocks(module: nn.Module):
    #     for module_name, module_child in module.named_children():
    #         if block_check_fn(module_child, module_name):
    #             blocks.append(module_child)
    #         else:
    #             _get_blocks(module_child)

    # # Run recursive function that updates the list blocks
    # _get_blocks(model)
    blocks = recurse_getattr(model, block_name)
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
        scale_optimizer_class: Optimizer = torch.optim.SGD,
        lr_scheduler_class: LRScheduler = LinearLR,
        optimizer_lr: float = 5e-3,
        batch_size: float = 8,
        learn_scale: bool = False,
        iters: int = 600,
        use_best_model: bool = True,
        use_amp: bool = True,
        amp_dtype: torch.dtype = torch.float16,
        optimizer_kwargs: Dict = {},
        lr_scheduler_kwargs: Dict = {
            "start_factor": 1.0,
            "end_factor": 0.0,
            "verbose": False,}
    ) -> None:
        if learned_round.iters != iters:
            warnings.warn(
                "The number of iterations passed to the learned round optimiser is different "
                "to that of the learned round method, which might lead to unexpected behaviour.")
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
        self.learn_scale = learn_scale
        self.scale_optimizer_class = scale_optimizer_class

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

    def _step(self, optimizer: List[Optimizer], lr_scheduler: List[LRScheduler]) -> None:
        for opt in optimizer:
            if opt:
                opt.step()
                opt.zero_grad()
        for sched in lr_scheduler:
            if sched:
                sched.step()

    def apply_learned_round(
            self,
            model: nn.Module,
            data_loader: DataLoader,
            block_name: Callable = None,
            keep_gpu: bool = True) -> None:
        self.learned_round._insert_learned_round_quantizer(model)

        # Prepare model for optimization
        self.learned_round_utils.init_model_learned_round(model)

        # block_check_fn = block_check_fn if block_check_fn else self.learned_round_utils.default_block_check_fn
        # Retrieve blocks using the appropiate function to check blocks
        blocks = get_blocks(model, block_name)

        print(f"Total Iterations per block {self.iters}")
        print(f"Number of blocks {len(blocks)}")

        # Initialise cache to store partial inputs and outputs for each block
        cache = self.learned_round_utils.init_cache()

        # Loop across blocks to optimise rounding within each
        for block_idx, (block, block_loss, block_learned_round_modules) in enumerate(
                self.learned_round.learned_round_iterator(blocks)):
            # Populate cache for the given block
            offload_model(model, gpu_device_map={0: "2GB"})

            n_samples = self.learned_round_utils.populate_cache(
                cache,
                model,
                block,
                data_loader,
                keep_gpu=keep_gpu,
            )
            remove_hooks(model)
            print("----")
            # offload_model(block)
            block.cuda()
            for params in block.parameters():
                params.requires_grad = False
            # Retrieve learned round modules
            learned_round_modules = self.learned_round._find_learned_round_modules(block)
            # Enable gradient tracking in learned round modules
            for round_module in learned_round_modules:
                round_module.train()
                for params in round_module.parameters():
                    params.requires_grad = True
            # Block needs to be in eval mode while the rounding is optimised
            block.eval()
            params = [
                list(learned_round_module.parameters())
                for learned_round_module in learned_round_modules]

            if self.learn_scale:
                p = []
                for m in block.modules():
                    if isinstance(m, WeightQuantProxyFromInjectorBase):
                        for n, v in m.named_parameters():
                            if n.endswith('scaling_impl.value'):
                                v.requires_grad = True
                                p.append(v)

                optimizer_scale = self.scale_optimizer_class(
                    p,
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

            # Initialise optimiser and LR scheduler
            optimizer = self.optimizer_class(
                itertools.chain.from_iterable(params),
                lr=self.optimizer_lr,
                **self.optimizer_kwargs,
            )
            lr_scheduler = (
                self.lr_scheduler_class(optimizer, **self.lr_scheduler_kwargs)
                if self.lr_scheduler_class else None)

            # Variables needed for printing
            best_loss = torch.finfo(torch.float).max
            init_loss = -1.0
            last_best_iter = self.iters

            optimal_rounding_params = {}

            torch.cuda.empty_cache()


            pbar = tqdm(range(self.iters), desc='')

            for i in pbar:
                # Sample mini-batch from cache
                idxs = torch.randperm(n_samples)[:self.batch_size]
                inputs, fp_outs = self.learned_round_utils.sample_cache(block, cache, idxs)

                # Run block forward to obtain quant outputs
                quant_outs = self.learned_round_utils.run_forward(block, inputs)

                if self.use_amp:
                    with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                  dtype=self.amp_dtype):
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
                self._step([optimizer, optimizer_scale], [lr_scheduler, lr_scheduler_scale])
                with torch.no_grad():
                    for pp in params:
                        pp[0].data = torch.clamp(
                            pp[0].data,
                            torch.tensor(-0.5).type_as(pp[0]),
                            torch.tensor(0.5).type_as(pp[0]))

                # Update progress bar
                pbar.set_description(
                    "Block = {:d}/{:d}, {}".format(
                        block_idx + 1,
                        len(blocks),
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
                f"initial loss: {init_loss:.6f}, best loss: {best_loss:.6f}, at iteration {last_best_iter}."
            )
        remove_hooks(block)
        # Finish optimisation
        self.learned_round_utils.finish_model_learned_round(model)
        for pp in params:
            print(torch.sum(torch.abs(pp[0]) > 0.5))
        # print(params)
