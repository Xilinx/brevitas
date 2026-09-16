# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
import torch.multiprocessing as mp
from torch.nn.utils import parametrize
from torch.utils.checkpoint import checkpoint

from brevitas.graph.equalize import fuse_parametrizations
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.optim.cailey_sgd import CaileySGD
from brevitas.utils.parametrization_utils import configure_rotation_input_materialization
from brevitas.utils.parametrization_utils import ensure_rotation_bank
from brevitas.utils.parametrization_utils import extract_trainable_rotation_matrix_owners
from brevitas.utils.parametrization_utils import get_rotation_bank
from brevitas.utils.parametrization_utils import reset_rotation_input_materialization_stats
from brevitas.utils.parametrization_utils import rotation_input_materialization_bytes
from brevitas.utils.parametrization_utils import RotationWeightParametrization
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.llm.llm_quant.fsdp_rotation import FSDPRotationCoordinator
from brevitas_examples.llm.llm_quant.fsdp_workarounds import enable_fsdp_unshard_sync


class RotationHolder(nn.Module):

    def __init__(self, rotation, group_id="r1"):
        super().__init__()
        self.rotation = RotationWeightParametrization(
            rotation, lambda tensor, matrix, K: tensor @ matrix, axis=1, rotation_group_id=group_id)


class RotationBlock(nn.Module):

    def __init__(self, rotation):
        super().__init__()
        self.first = RotationHolder(rotation)
        self.second = RotationHolder(rotation)
        self.linear = nn.Linear(2, 2)

    def forward(self, tensor):
        return self.linear(self.first.rotation(tensor) + self.second.rotation(tensor))


class RotationBankModel(nn.Module):

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


def rotation_coordinator(model, gradient_checkpointing=False, checkpoint_kwargs=None):
    plugin = SimpleNamespace(ignored_modules=None, cpu_ram_efficient_loading=False)
    accelerator = SimpleNamespace(
        state=SimpleNamespace(fsdp_plugin=plugin),
        device=torch.device("cpu"),
        clip_grad_norm_=nn.utils.clip_grad_norm_,
        unscale_gradients=lambda: None)
    args = SimpleNamespace(
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=checkpoint_kwargs)
    trainer = SimpleNamespace(accelerator=accelerator, args=args)
    return FSDPRotationCoordinator(trainer), plugin


def distributed_rotation_worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        model = RotationBankModel()
        with torch.no_grad():
            model.blocks[0].first.rotation.rot_mat.add_(rank)
        coordinator, plugin = rotation_coordinator(model)
        coordinator.prepare(model)
        bank = get_rotation_bank(model)
        rotation = bank.ordered_parameters()[0]
        assert torch.equal(rotation, torch.eye(2))

        ignored_parameters = set(bank.parameters())
        device_mesh = torch.distributed.device_mesh.init_device_mesh("cpu", (world_size,))
        for block in model.blocks:
            fully_shard(block, mesh=device_mesh, ignored_params=ignored_parameters)
        fully_shard(model, mesh=device_mesh, ignored_params=ignored_parameters)
        assert isinstance(model.blocks[0].linear.weight, DTensor)
        assert not isinstance(rotation, DTensor)

        value = torch.full((2, 2), rank + 1., requires_grad=True)
        model(value, checkpoint_blocks=True).sum().backward()
        local_gradient = rotation.grad.detach().clone()
        expected_gradient = local_gradient.clone()
        dist.all_reduce(expected_gradient)
        expected_gradient.div_(world_size)
        coordinator.consolidate_gradients()
        assert torch.allclose(rotation.grad, expected_gradient)
        grad_norm = coordinator.clip_grad_norm_(model.parameters(), max_norm=1.)
        assert torch.isfinite(grad_norm)

        optimizer = torch.optim.SGD([rotation], lr=0.1)
        optimizer.step()
        gathered = [torch.empty_like(rotation) for _ in range(world_size)]
        dist.all_gather(gathered, rotation)
        assert all(torch.equal(gathered[0], other) for other in gathered[1:])

        rotation.grad = torch.full_like(rotation, 6.) if rank == 1 else None
        coordinator.consolidate_gradients()
        assert torch.equal(rotation.grad, torch.full_like(rotation, 3.))

        cailey = CaileySGD([rotation], lr=0.01, stiefel=True)
        for _ in range(105):
            rotation.grad = torch.ones_like(rotation)
            cailey.step()
        dist.all_gather(gathered, rotation)
        assert all(torch.equal(gathered[0], other) for other in gathered[1:])

        coordinator.check_replica_consistency()
        with torch.no_grad():
            rotation.add_(rank)
        replica_divergence_detected = False
        try:
            coordinator.check_replica_consistency()
        except RuntimeError:
            replica_divergence_detected = True
        assert replica_divergence_detected
        with torch.no_grad():
            rotation.sub_(rank)

        from torch.distributed.checkpoint.state_dict import get_model_state_dict
        from torch.distributed.checkpoint.state_dict import StateDictOptions
        state_dict = get_model_state_dict(
            model,
            options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True, cpu_offload=True))
        if rank == 0:
            assert "_brevitas_rotation_bank.rotations.rotation_0000" in state_dict
            assert not isinstance(
                state_dict["_brevitas_rotation_bank.rotations.rotation_0000"], DTensor)
        else:
            assert not state_dict
    finally:
        dist.destroy_process_group()


def test_rotation_bank_owns_each_rotation_once():
    model = RotationBankModel()
    coordinator, plugin = rotation_coordinator(model)
    coordinator.prepare(model)

    bank = get_rotation_bank(model)
    rotation = bank.ordered_parameters()[0]
    consumers = [block.first.rotation for block in model.blocks] + [
        block.second.rotation for block in model.blocks]

    assert plugin.ignored_modules == [bank]
    assert extract_trainable_rotation_matrix_owners(model) == [rotation]
    assert all(consumer.rot_mat is rotation for consumer in consumers)
    assert all(not list(consumer.parameters(recurse=False)) for consumer in consumers)
    assert list(model.state_dict()).count("_brevitas_rotation_bank.rotations.rotation_0000") == 1

    model(torch.ones(2, 2, requires_grad=True)).sum().backward()
    assert rotation.grad is not None


def test_rotation_bank_deepcopy_rebinds_consumers():
    model = RotationBankModel()
    ensure_rotation_bank(model)

    copied = deepcopy(model)

    original_rotation = get_rotation_bank(model).ordered_parameters()[0]
    copied_rotation = get_rotation_bank(copied).ordered_parameters()[0]
    assert copied_rotation is not original_rotation
    assert all(block.first.rotation.rot_mat is copied_rotation for block in copied.blocks)
    assert all(block.second.rotation.rot_mat is copied_rotation for block in copied.blocks)


def test_rotation_bank_multiple_rotations_state_dict_round_trip():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.first = RotationHolder(nn.Parameter(torch.eye(2)), group_id="first")
            self.second = RotationHolder(nn.Parameter(2. * torch.eye(2)), group_id="second")

    model = Model()
    bank = ensure_rotation_bank(model)
    with torch.no_grad():
        bank.rotations["rotation_0000"].fill_(3.)
        bank.rotations["rotation_0001"].fill_(4.)
    state_dict = model.state_dict()
    restored = Model()
    ensure_rotation_bank(restored)

    restored.load_state_dict(state_dict, strict=True)

    assert list(bank.rotations) == ["rotation_0000", "rotation_0001"]
    assert list(state_dict) == [
        "_brevitas_rotation_bank.rotations.rotation_0000",
        "_brevitas_rotation_bank.rotations.rotation_0001"]
    assert torch.equal(restored.first.rotation.rot_mat, model.first.rotation.rot_mat)
    assert torch.equal(restored.second.rotation.rot_mat, model.second.rotation.rot_mat)


def test_rotation_bank_setup_is_atomic():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.first = RotationHolder(nn.Parameter(torch.eye(2)), group_id="shared")
            self.second = RotationHolder(nn.Parameter(2. * torch.eye(2)), group_id="shared")

    model = Model()

    with pytest.raises(RuntimeError, match="distinct parameters"):
        ensure_rotation_bank(model)

    assert get_rotation_bank(model) is None
    assert not model.first.rotation.is_rotation_bank_bound
    assert not model.second.rotation.is_rotation_bank_bound


def test_rotation_bank_rejects_one_parameter_in_multiple_groups():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            rotation = nn.Parameter(torch.eye(2))
            self.first = RotationHolder(rotation, group_id="first")
            self.second = RotationHolder(rotation, group_id="second")

    model = Model()

    with pytest.raises(RuntimeError, match="multiple logical groups"):
        ensure_rotation_bank(model)

    assert get_rotation_bank(model) is None


def test_fusing_rotation_parametrizations_removes_bank():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)
            rotation = nn.Parameter(torch.tensor([[0., -1.], [1., 0.]]))
            parametrize.register_parametrization(
                self.linear,
                "weight",
                RotationWeightParametrization(
                    rotation,
                    lambda tensor,
                    matrix,
                    K: tensor @ matrix,
                    axis=1,
                    rotation_group_id="r1"))

        def forward(self, value):
            return self.linear(value)

    model = Model()
    ensure_rotation_bank(model)
    value = torch.randn(2, 2)
    expected = model(value)

    fuse_parametrizations(model)

    assert get_rotation_bank(model) is None
    assert not parametrize.is_parametrized(model.linear)
    assert torch.equal(model(value), expected)


def test_rotation_bank_optimizer_contains_parameter_once():
    model = RotationBankModel()
    ensure_rotation_bank(model)
    parameters = extract_trainable_rotation_matrix_owners(model)

    optimizer = torch.optim.SGD(parameters, lr=0.1)

    assert len(optimizer.param_groups[0]["params"]) == 1
    assert optimizer.param_groups[0]["params"][0] is parameters[0]


def test_remove_hooks_clears_stale_device_map():
    model = nn.Linear(2, 2)
    model.hf_device_map = {"": 0, "_brevitas_rotation_bank": 0}

    remove_hooks(model)

    assert not hasattr(model, "hf_device_map")


def test_rotation_input_materialization_separates_storage_and_preserves_gradients():

    class Model(nn.Module):

        def __init__(self, materialize):
            super().__init__()
            rotation = nn.Parameter(torch.tensor([[0.8, -0.6], [0.6, 0.8]]))
            self.consumer = RotationWeightParametrization(
                rotation, lambda tensor, matrix, K: tensor @ matrix, axis=1, rotation_group_id="r1")
            ensure_rotation_bank(self)
            configure_rotation_input_materialization(self, materialize)

    reference = Model(materialize=False)
    materialized = Model(materialize=True)
    materialized.load_state_dict(reference.state_dict())
    reference_input = torch.randn(3, 2, requires_grad=True)
    materialized_input = reference_input.detach().clone().requires_grad_(True)
    materialized_storage = []

    def materialized_matmul(tensor, matrix, K):
        materialized_storage.append(tensor.data_ptr())
        return tensor @ matrix

    materialized.consumer.rot_func = materialized_matmul

    reference.consumer(reference_input).sum().backward()
    materialized.consumer(materialized_input).sum().backward()

    assert len(materialized_storage) == 1
    assert materialized_storage[0] != materialized_input.data_ptr()
    assert torch.equal(materialized_input.grad, reference_input.grad)
    assert torch.equal(
        get_rotation_bank(materialized).ordered_parameters()[0].grad,
        get_rotation_bank(reference).ordered_parameters()[0].grad)
    assert rotation_input_materialization_bytes(
        materialized) == materialized_input.numel() * materialized_input.element_size()
    assert materialized.consumer.materialized_input_calls == 1
    reset_rotation_input_materialization_stats(materialized)
    assert rotation_input_materialization_bytes(materialized) == 0
    assert materialized.consumer.materialized_input_calls == 0


def test_online_rotation_input_materialization_separates_storage_and_preserves_gradients():
    # The online activation rotation (RotatedModule.rotation_forward) is the second
    # rotation code path. Like the weight path, its clone must sever storage from the
    # FSDP2-unsharded input while preserving the forward output and input gradients.
    layer = nn.Linear(16, 8, bias=False)
    reference = RotatedModule(layer=deepcopy(layer))
    materialized = RotatedModule(layer=deepcopy(layer))

    configure_rotation_input_materialization(reference, False)
    configure_rotation_input_materialization(materialized, True)
    assert reference.materialize_input is False
    assert materialized.materialize_input is True

    reference_input = torch.randn(4, 16, requires_grad=True)
    materialized_input = reference_input.detach().clone().requires_grad_(True)

    reset_rotation_input_materialization_stats(materialized)
    reference_out = reference(reference_input)
    materialized_out = materialized(materialized_input)

    # The clone must be transparent to the forward result.
    assert torch.allclose(reference_out, materialized_out, atol=1e-6)

    # Storage independence: mutating the caller input's storage in place after the
    # forward must not alter the already-computed output, which is only true if the
    # rotation consumed a private clone rather than aliasing the caller input.
    materialized_out_detached = materialized_out.detach().clone()
    with torch.no_grad():
        materialized_input.add_(1000.0)
    assert torch.equal(materialized_out.detach(), materialized_out_detached)

    # Gradients still flow back to the original (pre-mutation) input graph node.
    materialized_out.sum().backward()
    reference_out.sum().backward()
    assert torch.equal(materialized_input.grad, reference_input.grad)

    expected_bytes = reference_input.numel() * reference_input.element_size()
    assert rotation_input_materialization_bytes(materialized) == expected_bytes
    assert materialized.materialized_input_calls == 1

    reset_rotation_input_materialization_stats(materialized)
    assert rotation_input_materialization_bytes(materialized) == 0
    assert materialized.materialized_input_calls == 0


def test_rotation_bank_reduces_all_gradients_with_one_collective(monkeypatch):
    model = RotationBankModel()
    model.extra_rotation = RotationHolder(nn.Parameter(2. * torch.eye(3)), group_id="r2")
    coordinator, _ = rotation_coordinator(model)
    coordinator.prepare(model)
    rotations = get_rotation_bank(model).ordered_parameters()
    for index, rotation in enumerate(rotations):
        rotation.grad = torch.full_like(rotation, index + 1.)
    calls = []

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def all_reduce(tensor, op=None):
        calls.append(tensor.numel())
        tensor.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", all_reduce)

    coordinator.consolidate_gradients()

    assert len(calls) == 1
    assert torch.equal(rotations[0].grad, torch.ones_like(rotations[0]))
    assert torch.equal(rotations[1].grad, torch.full_like(rotations[1], 2.))


def test_fsdp_unshard_sync_waits_for_real_unshard(monkeypatch):

    class FakeStream:

        def __init__(self):
            self.synchronize_count = 0

        def synchronize(self):
            self.synchronize_count += 1

    class FakeParamGroup:

        def __init__(self, stream):
            self._training_state = SimpleNamespace(name="PRE_BACKWARD")
            self._all_gather_result = object()
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

    # Selective pre-backward mode syncs only in the PRE_BACKWARD training state.
    assert enable_fsdp_unshard_sync(model, sync_pre_backward=True) == 1
    param_group.wait_for_unshard()
    assert stream.synchronize_count == 1
    param_group._training_state = SimpleNamespace(name="FORWARD")
    param_group.wait_for_unshard()
    assert stream.synchronize_count == 1
    # Re-enabling with forward sync updates the already-installed group in place.
    assert enable_fsdp_unshard_sync(model, sync_pre_backward=False, sync_forward=True) == 0
    param_group.wait_for_unshard()
    assert stream.synchronize_count == 2

    # No flags installs nothing.
    assert enable_fsdp_unshard_sync(model) == 0


def test_fsdp_unshard_sync_universal_syncs_in_every_state(monkeypatch):
    # The universal guard must synchronize after every unshard regardless of the
    # training state, which is what covers the backward checkpoint-recompute site
    # that the selective (state-gated) mode misses at larger model sizes.

    class FakeStream:

        def __init__(self):
            self.synchronize_count = 0

        def synchronize(self):
            self.synchronize_count += 1

    class FakeParamGroup:

        def __init__(self, stream):
            self._training_state = SimpleNamespace(name="FORWARD")
            self._all_gather_result = object()
            self.device_handle = SimpleNamespace(current_stream=lambda: stream)

        def wait_for_unshard(self):
            # Mirror FSDP2: a completed unshard clears the pending result. The
            # guard must read "pending" before this original call runs.
            self._all_gather_result = None

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

    assert enable_fsdp_unshard_sync(model, sync_all=True) == 1

    count = 0
    for state in ("FORWARD", "PRE_FORWARD", "PRE_BACKWARD", "IDLE", "UNKNOWN"):
        param_group._training_state = SimpleNamespace(name=state)
        param_group._all_gather_result = object()  # a fresh pending unshard
        param_group.wait_for_unshard()
        count += 1
        assert stream.synchronize_count == count, state

    # A no-op unshard (nothing pending) must not synchronize.
    param_group._all_gather_result = None
    param_group.wait_for_unshard()
    assert stream.synchronize_count == count


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_rotation_bank_supports_gradient_checkpointing(use_reentrant):
    model = RotationBankModel()
    coordinator, _ = rotation_coordinator(
        model,
        gradient_checkpointing=True,
        checkpoint_kwargs={"use_reentrant": use_reentrant})
    coordinator.prepare(model)

    assert model.input_grads_enabled is use_reentrant
    value = torch.ones(2, 2, requires_grad=True)
    for block in model.blocks:
        value = checkpoint(block, value, use_reentrant=use_reentrant)
    value.sum().backward()
    assert get_rotation_bank(model).ordered_parameters()[0].grad is not None


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_rotation_bank_reduces_across_ranks(tmp_path):
    init_file = tmp_path / "distributed_init"
    mp.spawn(distributed_rotation_worker, args=(2, str(init_file)), nprocs=2, join=True)
