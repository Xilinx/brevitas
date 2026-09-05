# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import sys
from types import ModuleType
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.tensor import DTensor
import torch.multiprocessing as mp
from torch.utils.checkpoint import checkpoint

from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrix_owners
from brevitas.utils.parametrization_utils import RotationWeightParametrization
from brevitas_examples.llm.llm_quant.fsdp_rotation import FSDPRotationCoordinator
from brevitas_examples.llm.llm_quant.fsdp_rotation import remove_rotation_aliases_from_optimizer
from brevitas_examples.llm.llm_quant.fsdp_workarounds import enable_fsdp_unshard_sync


class RotationHolder(nn.Module):

    def __init__(self, rotation):
        super().__init__()
        self.rotation = RotationWeightParametrization(
            rotation, lambda tensor, matrix, K: tensor @ matrix, axis=1, rotation_group_id="r1")


class RotationBlock(nn.Module):

    def __init__(self, rotation):
        super().__init__()
        self.first = RotationHolder(rotation)
        self.second = RotationHolder(rotation)
        self.linear = nn.Linear(2, 2)

    def forward(self, tensor):
        return self.linear(self.first.rotation(tensor) + self.second.rotation(tensor))


class RotationReplicaModel(nn.Module):

    def __init__(self):
        super().__init__()
        rotation = nn.Parameter(torch.eye(2))
        self.blocks = nn.ModuleList([RotationBlock(rotation), RotationBlock(rotation)])
        self.input_grads_enabled = False

    def enable_input_require_grads(self):
        self.input_grads_enabled = True

    def forward(self, tensor, checkpoint_blocks=False):
        for block in self.blocks:
            tensor = (
                checkpoint(block, tensor, use_reentrant=False)
                if checkpoint_blocks else block(tensor))
        return tensor


def rotation_coordinator(model, monkeypatch, gradient_checkpointing=False, checkpoint_kwargs=None):
    plugin = SimpleNamespace(
        ignored_modules=None,
        cpu_ram_efficient_loading=False,
        auto_wrap_policy=transformer_auto_wrap_policy)
    accelerator = SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=plugin),
        device=torch.device("cpu"),
        clip_grad_norm_=nn.utils.clip_grad_norm_,
        unscale_gradients=lambda: None)
    args = SimpleNamespace(
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=checkpoint_kwargs)
    trainer = SimpleNamespace(accelerator=accelerator, args=args)

    accelerate = ModuleType("accelerate")
    accelerate_utils = ModuleType("accelerate.utils")
    accelerate_fsdp_utils = ModuleType("accelerate.utils.fsdp_utils")
    accelerate_fsdp_utils.fsdp2_prepare_auto_wrap_policy = (
        lambda plugin, wrapped_model: lambda module: isinstance(module, RotationBlock))
    accelerate.utils = accelerate_utils
    accelerate_utils.fsdp_utils = accelerate_fsdp_utils
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)
    monkeypatch.setitem(sys.modules, "accelerate.utils", accelerate_utils)
    monkeypatch.setitem(sys.modules, "accelerate.utils.fsdp_utils", accelerate_fsdp_utils)
    return FSDPRotationCoordinator(trainer), plugin


def install_fake_accelerate():
    accelerate = ModuleType("accelerate")
    accelerate_utils = ModuleType("accelerate.utils")
    accelerate_fsdp_utils = ModuleType("accelerate.utils.fsdp_utils")
    accelerate_fsdp_utils.fsdp2_prepare_auto_wrap_policy = (
        lambda plugin, wrapped_model: lambda module: isinstance(module, RotationBlock))
    accelerate.utils = accelerate_utils
    accelerate_utils.fsdp_utils = accelerate_fsdp_utils
    sys.modules["accelerate"] = accelerate
    sys.modules["accelerate.utils"] = accelerate_utils
    sys.modules["accelerate.utils.fsdp_utils"] = accelerate_fsdp_utils


def distributed_rotation_worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        install_fake_accelerate()
        model = RotationReplicaModel()
        plugin = SimpleNamespace(
            ignored_modules=None,
            cpu_ram_efficient_loading=False,
            auto_wrap_policy=transformer_auto_wrap_policy)
        trainer = SimpleNamespace(
            accelerator=SimpleNamespace(
                state=SimpleNamespace(fsdp_plugin=plugin),
                device=torch.device("cpu"),
                clip_grad_norm_=nn.utils.clip_grad_norm_,
                unscale_gradients=lambda: None),
            args=SimpleNamespace(gradient_checkpointing=False, gradient_checkpointing_kwargs=None))
        coordinator = FSDPRotationCoordinator(trainer)
        coordinator.prepare(model)

        owner = extract_trainable_rotation_matrix_owners(model)[0]
        alias = model.blocks[1].first.rotation.rot_mat
        ignored_parameters = {
            parameter for module in plugin.ignored_modules for parameter in module.parameters()}
        device_mesh = torch.distributed.device_mesh.init_device_mesh("cpu", (world_size,))
        for block in model.blocks:
            fully_shard(block, mesh=device_mesh, ignored_params=ignored_parameters)
        fully_shard(model, mesh=device_mesh, ignored_params=ignored_parameters)
        assert isinstance(model.blocks[0].linear.weight, DTensor)
        assert not isinstance(owner, DTensor)
        assert not isinstance(alias, DTensor)

        value = torch.ones(2, 2, requires_grad=True)
        model(value, checkpoint_blocks=True).sum().backward()
        assert owner.grad is not None
        assert alias.grad is not None
        coordinator.consolidate_gradients()
        grad_norm = coordinator.clip_grad_norm_(model.parameters(), max_norm=1.)
        assert torch.isfinite(grad_norm)
        assert alias.grad is None
        model.zero_grad(set_to_none=True)

        owner.grad = torch.full_like(owner, 2.) if rank == 1 else None
        alias.grad = torch.full_like(alias, 4.) if rank == 1 else None
        coordinator.consolidate_gradients()
        assert torch.equal(owner.grad, torch.full_like(owner, 3.))
        assert alias.grad is None

        with torch.no_grad():
            owner.add_(rank)
        coordinator.synchronize_parameters(torch.optim.SGD([owner], lr=0.1))
        assert torch.equal(owner, alias)
        expected = torch.eye(2)
        assert torch.equal(owner, expected)
    finally:
        dist.destroy_process_group()


def test_fsdp_rotation_replicas_sum_gradients_and_update_together(monkeypatch):
    model = RotationReplicaModel()
    coordinator, plugin = rotation_coordinator(model, monkeypatch)
    coordinator.prepare(model)

    block_0_rotation = model.blocks[0].first.rotation.rot_mat
    block_1_rotation = model.blocks[1].first.rotation.rot_mat
    assert block_0_rotation is model.blocks[0].second.rotation.rot_mat
    assert block_1_rotation is model.blocks[1].second.rotation.rot_mat
    assert block_0_rotation is not block_1_rotation
    assert torch.equal(block_0_rotation, block_1_rotation)
    assert len(plugin.ignored_modules) == 4

    owners = extract_trainable_rotation_matrix_owners(model)
    assert len(owners) == 1
    assert owners[0] is block_0_rotation
    block_0_rotation.grad = torch.ones_like(block_0_rotation)
    block_1_rotation.grad = torch.full_like(block_1_rotation, 2.)
    coordinator.consolidate_gradients()
    assert torch.equal(block_0_rotation.grad, torch.full_like(block_0_rotation, 3.))
    assert block_1_rotation.grad is None
    grad_norm = nn.utils.clip_grad_norm_([block_0_rotation, block_1_rotation], max_norm=1.)
    assert grad_norm == pytest.approx(6.)

    optimizer = torch.optim.SGD(owners, lr=0.1)
    coordinator.attach_optimizer(optimizer)
    optimizer.step()
    assert torch.equal(block_0_rotation, block_1_rotation)


def test_fsdp_rotation_preserves_preexisting_optimizer_owner(monkeypatch):
    model = RotationReplicaModel()
    original_owner = extract_trainable_rotation_matrix_owners(model)[0]
    optimizer = torch.optim.SGD([original_owner], lr=0.1)
    coordinator, _ = rotation_coordinator(model, monkeypatch)

    coordinator.prepare(model)

    owner = extract_trainable_rotation_matrix_owners(model)[0]
    assert owner is original_owner
    assert len(optimizer.param_groups[0]["params"]) == 1
    assert optimizer.param_groups[0]["params"][0] is owner
    assert any(parameter is owner for parameter in model.parameters())


def test_fsdp_rotation_aliases_are_removed_from_generic_optimizer(monkeypatch):
    model = RotationReplicaModel()
    coordinator, _ = rotation_coordinator(model, monkeypatch)
    coordinator.prepare(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    remove_rotation_aliases_from_optimizer(optimizer)

    optimized_parameters = optimizer.param_groups[0]["params"]
    assert all(
        not getattr(parameter, '_brevitas_rotation_alias', False)
        for parameter in optimized_parameters)
    assert any(
        parameter is extract_trainable_rotation_matrix_owners(model)[0]
        for parameter in optimized_parameters)


def test_fsdp_rotation_aliases_follow_owner_trainability(monkeypatch):
    model = RotationReplicaModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    coordinator, _ = rotation_coordinator(model, monkeypatch)
    coordinator.prepare(model)
    owner = extract_trainable_rotation_matrix_owners(model)[0]
    owner.requires_grad_(True)
    optimizer = torch.optim.SGD([owner], lr=0.1)

    coordinator.attach_optimizer(optimizer)

    assert all(
        parameter.requires_grad for _,
        parameters in coordinator.replica_groups for parameter in parameters)


def test_fsdp_unshard_sync_waits_for_real_unshard(monkeypatch):

    class FakeStream:

        def __init__(self):
            self.synchronize_count = 0

        def synchronize(self):
            self.synchronize_count += 1

    class FakeParamGroup:

        def __init__(self, stream):
            self._training_state = SimpleNamespace(name="PRE_BACKWARD")
            self._module_fqn = "model.layers.3"
            self._all_gather_result = SimpleNamespace(
                all_gather_output=torch.empty(16, dtype=torch.bfloat16))
            self.device_handle = SimpleNamespace(current_stream=lambda: stream)

        def wait_for_unshard(self):
            return None

    class FakeFSDPModule(nn.Module):

        def __init__(self, param_group):
            super().__init__()
            self.state = SimpleNamespace(_fsdp_param_groups=[param_group])

        def _get_fsdp_state(self):
            return self.state

    import torch.distributed.fsdp as torch_fsdp

    stream = FakeStream()
    param_group = FakeParamGroup(stream)
    model = FakeFSDPModule(param_group)
    monkeypatch.setattr(torch_fsdp, "FSDPModule", FakeFSDPModule)

    assert enable_fsdp_unshard_sync(model) == 1
    assert enable_fsdp_unshard_sync(model) == 0
    param_group.wait_for_unshard()
    assert stream.synchronize_count == 1

    param_group._training_state = SimpleNamespace(name="FORWARD")
    param_group.wait_for_unshard()
    assert stream.synchronize_count == 1

    assert enable_fsdp_unshard_sync(model, sync_pre_backward=False, sync_forward=True) == 0
    param_group.wait_for_unshard()
    assert stream.synchronize_count == 2

    param_group._training_state = SimpleNamespace(name="PRE_BACKWARD")
    param_group._all_gather_result = None
    param_group.wait_for_unshard()
    assert stream.synchronize_count == 2

    forward_stream = FakeStream()
    forward_param_group = FakeParamGroup(forward_stream)
    forward_param_group._training_state = SimpleNamespace(name="FORWARD")
    forward_model = FakeFSDPModule(forward_param_group)

    assert enable_fsdp_unshard_sync(forward_model, sync_pre_backward=False, sync_forward=True) == 1
    forward_param_group.wait_for_unshard()
    assert forward_stream.synchronize_count == 1


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_fsdp_rotation_replicas_support_gradient_checkpointing(monkeypatch, use_reentrant):
    model = RotationReplicaModel()
    coordinator, _ = rotation_coordinator(
        model,
        monkeypatch,
        gradient_checkpointing=True,
        checkpoint_kwargs={"use_reentrant": use_reentrant})
    coordinator.prepare(model)

    assert model.input_grads_enabled is use_reentrant
    value = torch.ones(2, 2, requires_grad=True)
    for block in model.blocks:
        value = checkpoint(block, value, use_reentrant=use_reentrant)
    value.sum().backward()

    replicas = [model.blocks[index].first.rotation.rot_mat for index in range(2)]
    assert all(replica.grad is not None for replica in replicas)
    expected_gradient = sum(replica.grad.detach().clone() for replica in replicas)
    coordinator.consolidate_gradients()
    owner = extract_trainable_rotation_matrix_owners(model)[0]
    assert torch.equal(owner.grad, expected_gradient)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_fsdp_rotation_replicas_reduce_across_ranks(tmp_path):
    init_file = tmp_path / "distributed_init"
    mp.spawn(distributed_rotation_worker, args=(2, str(init_file)), nprocs=2, join=True)
