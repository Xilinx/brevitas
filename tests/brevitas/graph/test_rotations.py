# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from functools import partial
from functools import reduce
import itertools
from unittest.mock import patch

import pytest
import pytest_cases
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from brevitas.fx import symbolic_trace
from brevitas.graph.equalize import _apply_had_device
from brevitas.graph.equalize import _apply_ort_device
from brevitas.graph.equalize import _compute_rotations
from brevitas.graph.equalize import _get_input_axis
from brevitas.graph.equalize import _get_output_axis
from brevitas.graph.equalize import apply_rewriters
from brevitas.graph.equalize import EqualizationIndexes
from brevitas.graph.equalize import fuse_parametrizations
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import MergeLnAffine
from brevitas.graph.equalize import random_orthogonal_matrix
from brevitas.graph.equalize import Region
from brevitas.graph.hadamard import get_hadK
from brevitas.graph.quantize import LAYERWISE_COMPUTE_LAYER_MAP
from brevitas.graph.quantize import layerwise_quantize
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.utils.parametrization_utils import RotationWeightParametrization
from brevitas.utils.python_utils import recurse_getattr
from tests.marker import requires_pt_ge

from .equalization_fixtures import ATOL
from .equalization_fixtures import SEED
from .rotation_fixtures import *


@requires_pt_ge('2.4')
@pytest_cases.parametrize('partial_had', [True, False])
def test_models_rotation_fixtures(rotation_fixtures, partial_had):

    in_shape = (1, 4, 4)  # Special Attention shape

    model_class = rotation_fixtures
    model = model_class()
    inp = torch.ones(in_shape)

    model.eval()

    with torch.no_grad():
        expected_out = model(inp)

    model = symbolic_trace(model)
    merge = MergeLnAffine()
    model = merge.apply(model)
    eq = GraphRotationEqualization(
        orphan_sink=partial_had, return_rewriters=True, sdpa_regions=True)
    model, r = eq.apply(model)

    with torch.no_grad():
        out = model(inp)

    # Invariance of the output
    assert torch.allclose(out, expected_out, atol=ATOL)
    assert len(r) > 0


@pytest_cases.parametrize('N', [1, 2, 3], ids=lambda x: f"N={x}")
def test_composition_unfused_rotations(N):
    torch.manual_seed(SEED)

    for rotation_flags in itertools.product([False, True], repeat=N):

        in_features = 5
        module = nn.Linear(in_features=in_features, out_features=in_features)
        rot_module = copy.deepcopy(module)

        # Sample input to pass through the block
        sample_input = torch.rand((1, in_features),)
        # Composite rotation matrices
        rot_mat_input = torch.eye(in_features)
        rot_mat_output = torch.eye(in_features)

        for is_source in rotation_flags:
            # Generate a random matrix
            rot_mat = random_orthogonal_matrix(in_features).to(dtype=torch.float32)

            # Aggregate rotation matrices
            if is_source:
                rot_mat_output = rot_mat_output @ rot_mat
            else:
                rot_mat_input = rot_mat_input @ rot_mat

            # Compose rotation modules
            parametrize.register_parametrization(
                rot_module,
                "weight",
                RotationWeightParametrization(
                    rot_mat=rot_mat,
                    rot_func=_apply_ort_device,
                    axis=_get_output_axis(rot_module) if is_source else _get_input_axis(rot_module),
                ))
            if is_source:
                parametrize.register_parametrization(
                    rot_module,
                    "bias",
                    RotationWeightParametrization(
                        rot_mat=rot_mat,
                        rot_func=_apply_ort_device,
                        axis=1,
                    ))

        # If the node is a sink, the input is multiplied by the inverse of the rotation matrix x <- xQ^{-1}
        # If the node is a source, the output is multiplied by the rotation matrix o <- oQ
        gt_output = module(sample_input @ rot_mat_input.t()) @ rot_mat_output
        rot_output = rot_module(sample_input)

        # Verify that the rotation operations were computed correctly
        assert torch.allclose(gt_output, rot_output, atol=ATOL)


# Auxiliar method to convert a dictionary of sources/sinks into a valid region
def _instantiate_region(region_dict, model, expand_region=False) -> Region:
    if len(region_dict["srcs"]) > 0:
        sorted_srcs = dict(
            sorted({src: EqualizationIndexes(0, IN_FEATURES, 0) for src in region_dict["srcs"]
                   }.items()))
        sorted_sinks = dict(
            sorted({sink: EqualizationIndexes(0, IN_FEATURES, 0) for sink in region_dict["sinks"]
                   }.items()))
    else:
        sorted_srcs = dict()
        sorted_sinks = dict(
            sorted({sink: EqualizationIndexes(0, IN_FEATURES, 0) for sink in region_dict["sinks"]
                   }.items()))
    sorted_acts = tuple()
    expand_region = expand_region and len(region_dict["srcs"]) == 0
    return Region.from_dicts(
        srcs=sorted_srcs,
        sinks=sorted_sinks,
        acts=sorted_acts,
        name_to_module=model._modules,
        expand_region=expand_region)


# Auxiliar function to compare the weights of module instances belonging to classes_to_compare
def compare_model_weights(model_fused, model_unfused, classes_to_compare=(nn.Linear,)):
    tensor_names = ["weight", "bias"]
    for name_module_fused, module_fused in model_fused.named_modules():
        if isinstance(module_fused, classes_to_compare):
            module_unfused = reduce(getattr, [model_unfused] + name_module_fused.split("."))
            for tensor_name in tensor_names:
                if hasattr(module_fused, tensor_name) and getattr(module_fused,
                                                                  tensor_name) is not None:
                    assert torch.allclose(getattr(module_fused, tensor_name), getattr(module_unfused, tensor_name), atol=0.0, rtol=0.0), f"Tensor {tensor_name} does not match for module {name_module_fused}"


@requires_pt_ge('2.3.1')
@pytest_cases.parametrize(
    'mask',
    itertools.product([False, True], repeat=3),
    ids=lambda mask: "-".join([rot for mask_el, rot in zip(mask, ["R1", "R2", "R3"]) if mask_el]))
@pytest_cases.parametrize('full_rotation_method', ['ort', 'had'])
@pytest_cases.parametrize('device', ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
@pytest_cases.parametrize('fuse_rotations', [False, True], ids=["unfused", "fused"])
@pytest_cases.parametrize('use_fx', [True, False], ids=["fx", "no-fx"])
@pytest_cases.parametrize('expansion_step', [3, 0], ids=["expansion", "no-expansion"])
@pytest_cases.parametrize('rotation_block_size', [12, None])
@pytest_cases.parametrize('disable_block_rotation_for_fused', [True, False])
def test_compute_rotations(
        rotation_model,
        mask,
        full_rotation_method,
        device,
        fuse_rotations,
        use_fx,
        expansion_step,
        rotation_block_size,
        disable_block_rotation_for_fused):
    if expansion_step > 0 and full_rotation_method == 'ort':
        pytest.skip("Expansion is not compatible with orthogonal rotations")
    if rotation_block_size is not None and full_rotation_method == 'ort':
        pytest.skip("Block rotation is not compatible with orthogonal rotations")
    # Instantiate a residual model for which a collection of regions is available
    model = rotation_model()
    device = torch.device("cuda") if device == 'cuda' else torch.device("cpu")
    model.to(device)
    # Sample input to pass through the models
    sample_inputs = torch.rand(size=(5, IN_FEATURES)).to(device)
    # Collect only a subset of regions to be applied
    regions_dicts = [
        region_dict for mask_element,
        region_dict in zip(mask, RESIDUAL_MODEL_REGION_DICTS) if mask_element]
    # Use FX model if requested
    if use_fx:
        graph_model = symbolic_trace(model)
        # The module names in the original model need to be mapped to the ones
        # in graph_model
        map_model_graph = {}
        assigned_graph_modules = set()
        for graph_module_name, graph_module in graph_model.named_modules():
            if hasattr(graph_module, "weight"):
                for name, module in model.named_modules():
                    # The check name not in map_model_graph prevents the assignment to the same module
                    # when tied parameters are present
                    if name not in map_model_graph and graph_module_name not in assigned_graph_modules and hasattr(
                            module, "weight") and graph_module.weight is module.weight:
                        map_model_graph[name] = graph_module_name
                        assigned_graph_modules.add(graph_module_name)
        # Replace the names of the modules in sources/sinks by the names of the modules in the FX model
        regions_dicts = [{
            k: list(map(lambda x: map_model_graph[x], v))
            for k, v in region_dict.items()}
                         for region_dict in regions_dicts]
        # Rotation will be applied on the FX model
        model = graph_model

    # Deepcopy the models as parameters are going to be modified in-place
    rotated_model_unfused = copy.deepcopy(model)
    rotated_model_fused = copy.deepcopy(model)

    # Generator to control the random orthogonal matrices generated
    generator = torch.Generator()
    generator.manual_seed(SEED)
    # Clone generator to make sure we can use the same rotation matrices
    generator_clone = torch.Generator()
    generator_clone.set_state(generator.get_state())

    # Apply rotations on the model with unfused rotations
    expand_region = expansion_step > 1
    regions_unfused = list(
        map(
            lambda x: _instantiate_region(x, rotated_model_unfused, expand_region=expand_region),
            regions_dicts))

    def patched_function(tensor, had_K, K):
        rot_mat, K = get_hadK(had_K.shape[0])[0], get_hadK(had_K.shape[0])[1]
        return _apply_had_device(tensor, rot_mat, K)

    if full_rotation_method == 'had':
        # _apply_ort_device is patched to ensure that the hadamard matrices in hadamard.pt are used, instead of
        # the random ones generated by random_hadamard_matrices
        # we call apply_ort_device only if the dimension of rotation is not compatible with had
        with patch('brevitas.graph.equalize._apply_ort_device',
                   lambda tensor,
                   had_K,
                   K: patched_function(tensor, had_K, K)):
            rewriters = _compute_rotations(
                rotated_model_unfused,
                regions_unfused,
                full_rotation_method=full_rotation_method,
                fuse_rotations=False,
                expansion_step=expansion_step,
                rotation_block_size=rotation_block_size,
                disable_block_rotation_for_fused=disable_block_rotation_for_fused,
                generator=generator)
    elif full_rotation_method == 'ort':
        rewriters = _compute_rotations(
            rotated_model_unfused,
            regions_unfused,
            full_rotation_method=full_rotation_method,
            fuse_rotations=False,
            expansion_step=expansion_step,
            rotation_block_size=rotation_block_size,
            disable_block_rotation_for_fused=disable_block_rotation_for_fused,
            generator=generator)

    apply_rewriters(rotated_model_unfused, rewriters)

    # Apply rotations on the model with fused rotations
    regions_fused = list(
        map(
            lambda x: _instantiate_region(x, rotated_model_fused, expand_region=expand_region),
            regions_dicts))
    r = _compute_rotations(
        rotated_model_fused,
        regions_fused,
        full_rotation_method=full_rotation_method,
        fuse_rotations=True,
        expansion_step=expansion_step,
        rotation_block_size=rotation_block_size,
        disable_block_rotation_for_fused=disable_block_rotation_for_fused,
        generator=generator_clone)
    apply_rewriters(rotated_model_fused, r)

    # Compute outputs for each model
    model_output = model(sample_inputs)
    rotated_model_unfused_output = rotated_model_unfused(sample_inputs)
    rotated_model_fused_output = rotated_model_fused(sample_inputs)

    # Verify that the correct number of unique rotation matrices were included. Orphan sinks (len(region_dict["srcs"]) == 0) do not
    # an attached parametrization
    assert sum([len(region_dict["srcs"]) > 0 for region_dict in regions_dicts]) == sum([
        "rot_mat" in name for name,
        _ in rotated_model_unfused.named_parameters(remove_duplicate=True)])
    # Verify that RotatedModules were added appropiately
    for rotated_model in [rotated_model_fused, rotated_model_unfused]:
        assert sum([len(region_dict["srcs"]) == 0 for region_dict in regions_dicts]) == sum([
            isinstance(module, RotatedModule) for module in rotated_model.modules()])
    # Optionally fuse the rotations
    if fuse_rotations:
        rotated_model_unfused = fuse_parametrizations(rotated_model_unfused)
        # Verify that no parametrizations remain after fusing
        for module in rotated_model_unfused.modules():
            assert not parametrize.is_parametrized(module)
    # Outputs should match for rotated and unrotated models
    assert torch.allclose(model_output, rotated_model_fused_output, atol=ATOL)
    assert torch.allclose(
        rotated_model_unfused_output, rotated_model_fused_output, atol=0.0, rtol=0.0)
    # Verify that the weights have changed with respect to the unrotated module for the modules that have received parametrizations
    # Verify that weights match between the fused and unfused model
    compare_model_weights(rotated_model_fused, rotated_model_unfused)


@requires_pt_ge('2.3.1')
@pytest_cases.parametrize(
    'kwargs',
    [
        {
            'model': nn.Sequential(nn.Linear(2, 3)),
            'sample_input': torch.tensor([[0.8, -0.6]]),
            'rot_mat': torch.tensor([[1., -1.], [1., 1.]]) / torch.sqrt(torch.tensor(2.)),
            'rot_func': lambda tensor,
                        rot_mat,
                        K: torch.matmul(tensor, rot_mat),
            'key': '0',
            'expected': "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>"},])
def test_fuse_parametrized_modules(kwargs):
    key = kwargs['key']
    exp = kwargs['expected']
    rot_mat = kwargs['rot_mat']
    rot_func = kwargs['rot_func']
    model = kwargs["model"]
    sample_input = kwargs["sample_input"]
    module = recurse_getattr(model, key)
    # Register rotation parametrization to module
    parametrize.register_parametrization(
        module=module,
        tensor_name="weight",
        parametrization=RotationWeightParametrization(
            rot_mat=nn.Parameter(rot_mat),
            rot_func=rot_func,
            axis=1,
            K=None,
        ))
    compute_layer_map = copy.deepcopy(LAYERWISE_COMPUTE_LAYER_MAP)
    module = recurse_getattr(model, key)
    type_quant_module = parametrize.type_before_parametrizations(module)
    compute_layer_map[type_quant_module][1]["weight_quant"] = compute_layer_map[type_quant_module][
        1]["weight_quant"].let(scaling_impl_type='parameter_from_stats')
    qmodel = layerwise_quantize(model, compute_layer_map=compute_layer_map)
    # Calibration pass to initialize scales
    with torch.no_grad():
        output = qmodel(sample_input)
    # Fuse parametrizations
    qmodel = fuse_parametrizations(qmodel)
    # Verify that scales were not lost
    module = recurse_getattr(model, key)
    assert module.weight_quant.tensor_quant.scaling_impl.init_done
    assert not torch.allclose(
        module.weight_quant.tensor_quant.scaling_impl.value,
        torch.ones_like(module.weight_quant.tensor_quant.scaling_impl.value))
    # Compute output after fusing and check that it matches
    with torch.no_grad():
        output_fused = qmodel(sample_input)
    assert torch.allclose(output, output_fused, rtol=0.0, atol=0.0)


@requires_pt_ge('2.3.1')
@pytest_cases.parametrize('device', ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu'])
def test_fuse_parametrized_modules_after_compile_and_train(device):
    """Test that fuse_parametrizations works correctly after compile_quant + a training step.

    This specifically tests the fix where compile_quant adds '.orig_mod' keys to the
    state_dict, which need to be stripped before calling load_state_dict during
    fuse_parametrizations.
    """
    if device == 'cpu':
        pytest.skip('Compile tests are disabled on CPU')
    torch.set_default_device(device)

    model = nn.Sequential(nn.Linear(2, 3))
    sample_input = torch.tensor([[0.8, -0.6]])
    target = torch.tensor([[1.0, 0.0, -1.0]])
    rot_mat = torch.tensor([[1., -1.], [1., 1.]]) / torch.sqrt(torch.tensor(2.))
    rot_func = lambda tensor, rot_mat, K: torch.matmul(tensor, rot_mat)
    key = '0'

    module = recurse_getattr(model, key)
    # Register rotation parametrization to module
    parametrize.register_parametrization(
        module=module,
        tensor_name="weight",
        parametrization=RotationWeightParametrization(
            rot_mat=nn.Parameter(rot_mat),
            rot_func=rot_func,
            axis=1,
            K=None,
        ))
    compute_layer_map = copy.deepcopy(LAYERWISE_COMPUTE_LAYER_MAP)
    module = recurse_getattr(model, key)
    type_quant_module = parametrize.type_before_parametrizations(module)
    compute_layer_map[type_quant_module][1]["weight_quant"] = compute_layer_map[type_quant_module][
        1]["weight_quant"].let(scaling_impl_type='parameter_from_stats')
    qmodel = layerwise_quantize(model, compute_layer_map=compute_layer_map)

    # Calibration pass to initialize scales
    with torch.no_grad():
        output_pre_compile = qmodel(sample_input)

    # Compile quant proxies (this causes '.orig_mod' keys in state_dict)
    module = recurse_getattr(qmodel, key)
    for submodule in module.modules():
        if hasattr(submodule, 'compile_quant'):
            submodule.compile_quant()

    # Verify that compile was applied
    assert module.weight_quant.is_proxy_compiled

    # Fake training loop: single element forward + backward + optimizer step
    optimizer = torch.optim.SGD(qmodel.parameters(), lr=1e-4)
    qmodel.train()
    optimizer.zero_grad()
    train_output = qmodel(sample_input)
    loss = torch.nn.functional.mse_loss(train_output, target)
    loss.backward()
    optimizer.step()

    # Get output before fusing (after the training step)
    qmodel.eval()
    with torch.no_grad():
        output_before_fuse = qmodel(sample_input)

    # Verify that scales were initialized and are non-trivial
    module = recurse_getattr(qmodel, key)
    assert module.weight_quant.tensor_quant.scaling_impl.init_done
    assert not torch.allclose(
        module.weight_quant.tensor_quant.scaling_impl.value,
        torch.ones_like(module.weight_quant.tensor_quant.scaling_impl.value))

    # Fuse parametrizations — this is where the fix for '.orig_mod' keys matters
    qmodel = fuse_parametrizations(qmodel)

    # Verify that no parametrizations remain after fusing
    for mod in qmodel.modules():
        assert not parametrize.is_parametrized(mod)

    # Verify that scales are still initialized after fuse
    module = recurse_getattr(qmodel, key)
    assert module.weight_quant.tensor_quant.scaling_impl.init_done
    assert not torch.allclose(
        module.weight_quant.tensor_quant.scaling_impl.value,
        torch.ones_like(module.weight_quant.tensor_quant.scaling_impl.value))

    # Compute output after fusing and check that it matches
    with torch.no_grad():
        output_after_fuse = qmodel(sample_input)
    assert torch.allclose(output_before_fuse, output_after_fuse, rtol=0.0, atol=0.0)
    torch.set_default_device('cpu')
