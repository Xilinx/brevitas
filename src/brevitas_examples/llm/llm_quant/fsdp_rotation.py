# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import math
import re

import torch

from brevitas.utils.parametrization_utils import get_rotation_groups
from brevitas.utils.parametrization_utils import RotationWeightParametrization


def remove_rotation_aliases_from_optimizer(optimizer: torch.optim.Optimizer) -> None:
    """Remove physical rotation aliases while retaining each logical owner."""
    optimizers = [optimizer, *getattr(optimizer, 'optimizers', [])]
    visited = set()
    for current_optimizer in optimizers:
        if id(current_optimizer) in visited:
            continue
        visited.add(id(current_optimizer))
        for group in current_optimizer.param_groups:
            group["params"] = [
                parameter for parameter in group["params"]
                if not getattr(parameter, '_brevitas_rotation_alias', False)]


class FSDPRotationCoordinator:
    """Keep per-FSDP-unit rotation replicas equivalent to one shared parameter."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.prepared = False
        self.replica_groups = []
        self._optimizer_hook = None
        self._original_clip_grad_norm = None

    @staticmethod
    def _rotation_unit(module_name, wrapped_module_names):
        candidates = [
            name for name in wrapped_module_names
            if module_name == name or module_name.startswith(name + ".")]
        if candidates:
            return max(candidates, key=len)
        # Embeddings and output heads may be separately wrapped by FSDP2 even though
        # they are not selected by the transformer auto-wrap policy.
        return module_name.split(".parametrizations.", 1)[0]

    def prepare(self, model: torch.nn.Module) -> None:
        if self.prepared:
            return
        from accelerate.utils.fsdp_utils import fsdp2_prepare_auto_wrap_policy

        plugin = self.trainer.accelerator.state.fsdp_plugin
        if plugin.cpu_ram_efficient_loading:
            raise RuntimeError(
                "FSDP2 CPU-RAM-efficient loading is not supported with replicated rotations.")
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        configured_policy = plugin.auto_wrap_policy
        if isinstance(configured_policy, functools.partial):
            configured_policy = configured_policy.func
        if configured_policy is not transformer_auto_wrap_policy:
            raise RuntimeError(
                "FSDP2 rotation replication requires a transformer-based auto-wrap policy.")
        auto_wrap_policy = fsdp2_prepare_auto_wrap_policy(plugin, model)
        if auto_wrap_policy is None:
            raise RuntimeError(
                "FSDP2 rotation replication could not resolve the transformer auto-wrap policy.")

        named_modules = list(model.named_modules())
        wrapped_module_names = [
            name for name, module in named_modules if name and auto_wrap_policy(module)]
        if not wrapped_module_names:
            raise RuntimeError("The FSDP2 auto-wrap policy did not select any model modules.")

        occurrences_by_group = {}
        for name, module in named_modules:
            if isinstance(module, RotationWeightParametrization):
                group_id = getattr(module, 'rotation_group_id', None)
                group_id = group_id if group_id is not None else ("legacy", id(module.rot_mat))
                if getattr(module, 'rotation_group_id', None) is None:
                    module.rotation_group_id = group_id
                occurrences_by_group.setdefault(group_id, []).append((name, module))

        for occurrences in occurrences_by_group.values():
            replicas = {}
            original_owner = occurrences[0][1].rot_mat
            # FSDP does not move ignored parameters, so place the preserved owner
            # before optimizers capture it and clone aliases on the same device.
            original_owner.data = original_owner.data.to(self.trainer.accelerator.device)
            for name, module in occurrences:
                unit = self._rotation_unit(name, wrapped_module_names)
                if unit not in replicas:
                    replicas[unit] = (
                        original_owner if not replicas else torch.nn.Parameter(
                            original_owner.detach().clone(),
                            requires_grad=original_owner.requires_grad))
                module.rot_mat = replicas[unit]

            owner = original_owner
            owner._brevitas_rotation_alias = False
            for replica in replicas.values():
                if replica is not owner:
                    replica._brevitas_rotation_alias = True
            for _, module in occurrences:
                module.rotation_is_owner = module.rot_mat is owner

        groups = get_rotation_groups(model)
        for modules in groups.values():
            parameters = []
            owner = None
            for module in modules:
                parameter = module.rot_mat
                if all(parameter is not existing for existing in parameters):
                    parameters.append(parameter)
                if getattr(module, 'rotation_is_owner', False):
                    owner = parameter
            if owner is not None:
                self.replica_groups.append((owner, parameters))

        rotation_modules = [module for modules in groups.values() for module in modules]
        ignored_modules = plugin.ignored_modules
        if isinstance(ignored_modules, str):
            pattern = re.compile(ignored_modules)
            ignored_modules = [module for name, module in named_modules if pattern.fullmatch(name)]
        else:
            ignored_modules = list(ignored_modules or [])
        plugin.ignored_modules = list(dict.fromkeys(ignored_modules + rotation_modules))

        if self.trainer.args.gradient_checkpointing:
            checkpoint_kwargs = self.trainer.args.gradient_checkpointing_kwargs or {}
            if checkpoint_kwargs.get("use_reentrant", True):
                if not hasattr(model, "enable_input_require_grads"):
                    raise RuntimeError(
                        "Reentrant gradient checkpointing with rotation training requires "
                        "model.enable_input_require_grads(). Use non-reentrant checkpointing "
                        "instead.")
                model.enable_input_require_grads()
        self._original_clip_grad_norm = self.trainer.accelerator.clip_grad_norm_
        self.trainer.accelerator.clip_grad_norm_ = self.clip_grad_norm_
        self.prepared = True

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        """Clip a mixed collection of FSDP DTensors and replicated rotations."""
        from torch.distributed.tensor import DTensor

        parameters = list(parameters)
        dtensor_parameters = [
            parameter for parameter in parameters
            if isinstance(parameter, DTensor) and parameter.grad is not None]
        tensor_parameters = [
            parameter for parameter in parameters
            if not isinstance(parameter, DTensor) and parameter.grad is not None]
        if not dtensor_parameters or not tensor_parameters:
            return self._original_clip_grad_norm(parameters, max_norm, norm_type)

        self.trainer.accelerator.unscale_gradients()
        dtensor_norm = torch.nn.utils.clip_grad_norm_(
            dtensor_parameters, float('inf'), norm_type=norm_type)
        if isinstance(dtensor_norm, DTensor):
            dtensor_norm = dtensor_norm.full_tensor()
        tensor_norm = torch.nn.utils.clip_grad_norm_(
            tensor_parameters, float('inf'), norm_type=norm_type)

        dtensor_norm_value = float(dtensor_norm)
        tensor_norm_value = float(tensor_norm)
        if math.isinf(float(norm_type)):
            total_norm_value = max(dtensor_norm_value, tensor_norm_value)
        else:
            total_norm_value = (
                dtensor_norm_value ** float(norm_type) +
                tensor_norm_value ** float(norm_type)) ** (1. / float(norm_type))
        clip_coefficient = min(float(max_norm) / (total_norm_value + 1e-6), 1.)
        for parameter in dtensor_parameters + tensor_parameters:
            parameter.grad.mul_(clip_coefficient)
        return torch.tensor(total_norm_value, device=tensor_parameters[0].device)

    def consolidate_gradients(self) -> None:
        if not self.prepared:
            return
        import torch.distributed as dist

        for owner, parameters in self.replica_groups:
            gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
            if dist.is_initialized():
                has_gradient = torch.tensor(bool(gradients), dtype=torch.int32, device=owner.device)
                dist.all_reduce(has_gradient, op=dist.ReduceOp.MAX)
                if not has_gradient.item():
                    continue
            elif not gradients:
                continue
            gradient = (
                torch.zeros_like(owner) if owner.grad is None else owner.grad.detach().clone())
            for parameter in parameters:
                if parameter is not owner and parameter.grad is not None:
                    gradient.add_(parameter.grad)
                if parameter is not owner:
                    parameter.grad = None
            if dist.is_initialized():
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                gradient.div_(dist.get_world_size())
            owner.grad = gradient

    def synchronize_parameters(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.prepared:
            return
        import torch.distributed as dist

        optimizers = getattr(optimizer, 'optimizers', [optimizer])
        with torch.no_grad():
            for owner, parameters in self.replica_groups:
                if dist.is_initialized():
                    dist.broadcast(owner, src=0)
                for parameter in parameters:
                    if parameter is not owner:
                        parameter.copy_(owner)
                if owner.is_cuda:
                    torch.cuda.current_stream(owner.device).synchronize()
                for sub_optimizer in optimizers:
                    state = sub_optimizer.state.get(owner, {})
                    for value in state.values():
                        if (torch.is_tensor(value) and value.device == owner.device and
                                dist.is_initialized()):
                            dist.broadcast(value, src=0)

    def attach_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        # Parameter selectors run after coordinator preparation and may re-enable only
        # the logical owner. Aliases need gradients even though they are not optimized.
        for owner, parameters in self.replica_groups:
            for parameter in parameters:
                parameter.requires_grad_(owner.requires_grad)
        if self._optimizer_hook is None:
            self._optimizer_hook = optimizer.register_step_post_hook(
                lambda current_optimizer,
                args,
                kwargs: self.synchronize_parameters(current_optimizer))
