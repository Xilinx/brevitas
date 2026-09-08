# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
import re

import torch

from brevitas.utils.parametrization_utils import ensure_rotation_bank


class FSDPRotationCoordinator:
    """Coordinate a root-owned, FSDP-ignored bank of replicated rotations."""

    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self.prepared = False
        self.bank = None
        self._original_clip_grad_norm = None

    def prepare(self, model: torch.nn.Module) -> None:
        if self.prepared:
            return

        plugin = self.trainer.accelerator.state.fsdp_plugin
        if plugin.cpu_ram_efficient_loading:
            raise RuntimeError(
                "FSDP2 CPU-RAM-efficient loading is not supported with replicated rotations.")

        self.bank = ensure_rotation_bank(model)
        if self.bank is None:
            raise RuntimeError("FSDP rotation training requires a non-empty RotationBank.")
        self.bank.to(self.trainer.accelerator.device)

        named_modules = list(model.named_modules())
        ignored_modules = plugin.ignored_modules
        if isinstance(ignored_modules, str):
            pattern = re.compile(ignored_modules)
            ignored_modules = [module for name, module in named_modules if pattern.fullmatch(name)]
        else:
            ignored_modules = list(ignored_modules or [])
        plugin.ignored_modules = list(dict.fromkeys(ignored_modules + [self.bank]))

        # The bank is ignored by FSDP, so synchronize its initial value explicitly.
        import torch.distributed as dist
        if dist.is_initialized():
            parameters = self.bank.ordered_parameters()
            flat_parameters = torch.nn.utils.parameters_to_vector(parameters)
            dist.broadcast(flat_parameters, src=0)
            torch.nn.utils.vector_to_parameters(flat_parameters, parameters)

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
        """Average every bank gradient with one fixed-order distributed collective."""
        if not self.prepared:
            return
        import torch.distributed as dist

        parameters = self.bank.ordered_parameters()
        if not parameters or not dist.is_initialized():
            return
        first = parameters[0]
        if any(parameter.device != first.device or parameter.dtype != first.dtype
               for parameter in parameters):
            raise RuntimeError(
                "RotationBank gradient reduction requires one device and dtype per bank.")

        gradient_parts = [
            torch.zeros_like(parameter).reshape(-1)
            if parameter.grad is None else parameter.grad.detach().reshape(-1)
            for parameter in parameters]
        presence = torch.tensor([parameter.grad is not None for parameter in parameters],
                                dtype=first.dtype,
                                device=first.device)
        gradient_numel = sum(part.numel() for part in gradient_parts)
        packed = torch.cat([*gradient_parts, presence])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)

        world_size = dist.get_world_size()
        offset = 0
        global_presence = packed[gradient_numel:]
        for index, parameter in enumerate(parameters):
            next_offset = offset + parameter.numel()
            if global_presence[index].item() == 0:
                parameter.grad = None
            else:
                parameter.grad = packed[offset:next_offset].view_as(parameter).div(world_size)
            offset = next_offset

    def check_replica_consistency(self) -> None:
        """Assert every rank holds an identical copy of each bank parameter."""
        if not self.prepared:
            return
        import torch.distributed as dist

        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        with torch.no_grad():
            for index, parameter in enumerate(self.bank.ordered_parameters()):
                gathered = [torch.empty_like(parameter) for _ in range(world_size)]
                dist.all_gather(gathered, parameter.contiguous())
                for rank, other in enumerate(gathered[1:], start=1):
                    if not torch.equal(gathered[0], other):
                        max_difference = (gathered[0] - other).abs().max().item()
                        raise RuntimeError(
                            f"RotationBank parameter {index} diverged on rank {rank}; "
                            f"max abs difference {max_difference}.")
