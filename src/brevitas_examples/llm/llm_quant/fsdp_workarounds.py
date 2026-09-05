# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools

import torch


def enable_fsdp_unshard_sync(
        model: torch.nn.Module, sync_pre_backward: bool = True, sync_forward: bool = False) -> int:
    """Synchronize selected FSDP2 unshards before their parameters are consumed."""
    if not sync_pre_backward and not sync_forward:
        return 0
    from torch.distributed.fsdp import FSDPModule

    installed = 0
    visited_param_groups = set()
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        if not hasattr(module, "_get_fsdp_state"):
            raise RuntimeError(
                "The installed PyTorch FSDPModule does not expose _get_fsdp_state().")
        state = module._get_fsdp_state()
        if not hasattr(state, "_fsdp_param_groups"):
            raise RuntimeError(
                "The installed PyTorch FSDP2 state does not expose _fsdp_param_groups.")
        for param_group in state._fsdp_param_groups:
            if id(param_group) in visited_param_groups:
                continue
            visited_param_groups.add(id(param_group))
            if getattr(param_group, "_brevitas_unshard_sync_installed", False):
                param_group._brevitas_sync_pre_backward |= sync_pre_backward
                param_group._brevitas_sync_forward |= sync_forward
                continue
            required_attributes = (
                "wait_for_unshard", "_training_state", "_all_gather_result", "device_handle")
            missing_attributes = [
                name for name in required_attributes if not hasattr(param_group, name)]
            if missing_attributes:
                raise RuntimeError(
                    "The installed PyTorch FSDP2 parameter-group implementation is not "
                    "compatible with FSDP unshard synchronization; missing attributes: "
                    f"{missing_attributes}.")
            original_wait_for_unshard = param_group.wait_for_unshard
            param_group._brevitas_sync_pre_backward = sync_pre_backward
            param_group._brevitas_sync_forward = sync_forward

            @functools.wraps(original_wait_for_unshard)
            def synchronized_wait_for_unshard(
                    *args,
                    _param_group=param_group,
                    _original_wait_for_unshard=original_wait_for_unshard,
                    **kwargs):
                training_state = getattr(_param_group, "_training_state", None)
                training_state = getattr(training_state, "name", str(training_state))
                pending = getattr(_param_group, "_all_gather_result", None) is not None
                result = _original_wait_for_unshard(*args, **kwargs)
                should_sync = pending and (
                    (_param_group._brevitas_sync_pre_backward and training_state == "PRE_BACKWARD")
                    or (_param_group._brevitas_sync_forward and training_state == "FORWARD"))
                if should_sync:
                    _param_group.device_handle.current_stream().synchronize()
                return result

            param_group.wait_for_unshard = synchronized_wait_for_unshard
            param_group._brevitas_unshard_sync_installed = True
            installed += 1

    if not visited_param_groups:
        raise RuntimeError(
            "FSDP unshard synchronization is enabled, but the prepared model has no "
            "FSDP2 parameter groups.")
    return installed
