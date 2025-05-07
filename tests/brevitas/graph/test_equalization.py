# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
from functools import partial
from functools import reduce
import itertools
from unittest.mock import patch

from packaging.version import parse
import pytest
import torch
import torch.nn.utils.parametrize as parametrize
from torchvision import models

from brevitas import torch_version
from brevitas.fx import symbolic_trace
from brevitas.graph.base import ModuleInstanceRegisterParametrization
from brevitas.graph.equalize import _apply_had_device
from brevitas.graph.equalize import _apply_ort_device
from brevitas.graph.equalize import _apply_rotate
from brevitas.graph.equalize import _batch_norm
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _get_input_axis
from brevitas.graph.equalize import _get_output_axis
from brevitas.graph.equalize import _is_supported_module
from brevitas.graph.equalize import _supported_layers
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.equalize import EqualizationIndexes
from brevitas.graph.equalize import fuse_parametrizations
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import MergeLnAffine
from brevitas.graph.equalize import random_orthogonal_matrix
from brevitas.graph.equalize import Region
from brevitas.graph.hadamard import get_hadK
from brevitas.graph.quantize import LAYERWISE_COMPUTE_LAYER_MAP
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.graph.utils import get_module
from brevitas.nn.equalized_layer import RotatedModule
from brevitas.utils.parametrization_utils import RotationWeightParametrization
from brevitas.utils.python_utils import recurse_getattr
from tests.marker import requires_pt_ge

from .equalization_fixtures import *


def test_resnet18_equalization():
    model = models.resnet18(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()
    model = symbolic_trace(model)
    expected_out = model(inp)

    model_orig = copy.deepcopy(model)
    supported_sinks = list(_supported_layers)
    supported_sinks = tuple([
        x for x in _supported_layers if x not in (torch.nn.LayerNorm, *_batch_norm)])
    regions = _extract_regions(model, state_impl_kwargs={'supported_sinks': supported_sinks})
    _ = equalize_test(
        model, regions, merge_bias=True, bias_shrinkage='vaiq', scale_computation_type='maxabs')
    out = model(inp)

    regions = sorted(regions, key=lambda region: sorted([r for r in region.srcs_names]))
    resnet_18_regions = sorted(RESNET_18_REGIONS, key=lambda region: region[0][0])
    equalized_layers = set()
    for r in resnet_18_regions:
        equalized_layers.update(r[0])
        equalized_layers.update(r[1])

    # Check that we found all the expected regions
    for region, expected_region in zip(regions, resnet_18_regions):
        srcs = region.srcs_names
        sources_check = set(srcs) == set(expected_region[0])
        sinks = region.sinks_names
        sinks_check = set(sinks) == set(expected_region[1])
        assert sources_check
        assert sinks_check

    # Check that all layers were equalized and weights changed
    for layer in equalized_layers:
        eq_module = get_module(model, layer)
        orig_module = get_module(model_orig, layer)
        assert not torch.allclose(eq_module.weight, orig_module.weight)

    # Check that equalization is not introducing FP variations
    assert torch.allclose(expected_out, out, atol=ATOL)


@pytest_cases.parametrize("merge_bias", [True, False])
def test_equalization_torchvision_models(model_coverage: tuple, merge_bias: bool):
    model, coverage = model_coverage

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()
    # The isistance does not work after symbolic trace
    is_alexnet = isinstance(model, models.AlexNet)
    model = symbolic_trace(model)
    model = TorchFunctionalToModule().apply(model)

    expected_out = model(inp)

    supported_sinks = list(_supported_layers)
    supported_sinks = tuple([
        x for x in _supported_layers if x not in (torch.nn.LayerNorm, *_batch_norm)])
    regions = _extract_regions(model, state_impl_kwargs={'supported_sinks': supported_sinks})
    scale_factor_regions = equalize_test(
        model,
        regions,
        merge_bias=merge_bias,
        bias_shrinkage='vaiq',
        scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    srcs = set()
    sinks = set()
    for r in regions:
        srcs.update([x for x in list(r.srcs_names)])
        sinks.update([x for x in list(r.sinks_names)])

    count_region_srcs = 0
    count_region_sinks = 0
    for n in model.graph.nodes:
        if _is_supported_module(model, n):
            count_region_srcs += 1
            if not isinstance(get_module(model, n.target), (nn.LayerNorm,) + _batch_norm):
                count_region_sinks += 1

    src_coverage = len(srcs) / count_region_srcs
    sink_coverage = len(sinks) / count_region_sinks
    assert src_coverage >= coverage[0]
    assert sink_coverage >= coverage[1]
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Graph equalization can exit in case of shape mismatches or other error without performing any
    # equalization and returning a scalar value. We check that the equalized regions are as many as
    # expected
    if is_alexnet:
        # In AlexNet, we cannot equalize only through one region
        assert sum([shape == () for shape in shape_scale_regions]) == 1
    else:
        assert all([shape != () for shape in shape_scale_regions])


@pytest_cases.parametrize("merge_bias", [True, False])
def test_models(toy_model, merge_bias, request):
    test_id = request.node.callspec.id

    if 'mha' in test_id:
        in_shape = IN_SIZE_LINEAR
    else:
        in_shape = IN_SIZE_CONV

    model_class = toy_model
    model = model_class()
    inp = torch.randn(in_shape)

    model.eval()
    with torch.no_grad():
        expected_out = model(inp)

    model = symbolic_trace(model)
    supported_sinks = list(_supported_layers)
    supported_sinks = tuple([
        x for x in _supported_layers if x not in (torch.nn.LayerNorm, *_batch_norm)])
    regions = _extract_regions(model, state_impl_kwargs={'supported_sinks': supported_sinks})
    scale_factor_regions = equalize_test(
        model,
        regions,
        merge_bias=merge_bias,
        bias_shrinkage='vaiq',
        scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    with torch.no_grad():
        out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    if 'convgroupconv' in test_id:
        with pytest.raises(AssertionError):
            assert all([shape != () for shape in shape_scale_regions])
    else:
        assert all([shape != () for shape in shape_scale_regions])


@pytest_cases.parametrize("layerwise", [True, False])
@pytest_cases.parametrize("fuse_scaling", [True, False])
@pytest_cases.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16],
    ids=lambda dtype: str(dtype).split(".")[-1])
@pytest_cases.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    ids=lambda dtype: str(dtype).split(".")[-1])
def test_act_equalization_models(toy_model, layerwise, fuse_scaling, dtype, device, request):
    if not fuse_scaling and parse('1.9.0') > torch_version:
        pytest.skip("Parametrizations were not available in PyTorch versions below 1.9.0")
    if dtype in [torch.float16, torch.bfloat16] and parse('2.3.0') > torch_version:
        pytest.skip(
            "Some operations are not implemented for float16/bfloat16 in PyTorch versions below 2.3.0"
        )
    test_id = request.node.callspec.id

    if 'mha' in test_id:
        in_shape = IN_SIZE_LINEAR
    else:
        in_shape = IN_SIZE_CONV

    model_class = toy_model
    model = model_class()
    model.to(device=device, dtype=dtype)
    inp = torch.randn(in_shape, device=device, dtype=dtype)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)
    with torch.no_grad():
        with activation_equalization_mode(model,
                                          0.5,
                                          True,
                                          layerwise=layerwise,
                                          fuse_scaling=fuse_scaling) as aem:
            regions = aem.graph_act_eq.regions
            model(inp)
    scale_factor_regions = aem.scale_factors
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL_DICT[dtype])

    # This region is made up of a residual branch, so no regions are found for act equalization
    if 'convgroupconv' in test_id:
        with pytest.raises(AssertionError):
            assert len(regions) > 0
            # Check that at least one region performs "true" equalization
            # If all shapes are scalar, no equalization has been performed
            assert all([shape != () for shape in shape_scale_regions])
    else:
        assert len(regions) > 0
        # Check that at least one region performs "true" equalization
        # If all shapes are scalar, no equalization has been performed
        assert all([shape != () for shape in shape_scale_regions])


@pytest_cases.parametrize(
    "model_dict", [(model_name, coverage) for model_name, coverage in MODELS.items()],
    ids=[model_name for model_name, _ in MODELS.items()])
@pytest_cases.parametrize("layerwise", [True, False])
@pytest_cases.parametrize("fuse_scaling", [True, False])
def test_act_equalization_torchvision_models(model_dict: dict, layerwise: bool, fuse_scaling: bool):
    if not fuse_scaling and parse('1.9.0') > torch_version:
        pytest.skip("Parametrizations were not available in PyTorch versions below 1.9.0")
    model, coverage = model_dict

    if model == 'googlenet' and torch_version == version.parse('1.8.1'):
        pytest.skip(
            'Skip because of PyTorch error = AttributeError: \'function\' object has no attribute \'GoogLeNetOutputs\' '
        )
    if 'vit' in model and torch_version < version.parse('1.13'):
        pytest.skip(
            f'ViT supported from torch version 1.13, current torch version is {torch_version}')

    try:
        model = getattr(models, model)(pretrained=True, transform_input=False)
    except TypeError:
        model = getattr(models, model)(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()

    model = symbolic_trace(model)
    model = TorchFunctionalToModule().apply(model)
    model = DuplicateSharedStatelessModule().apply(model)
    expected_out = model(inp)

    with torch.no_grad():
        with activation_equalization_mode(model,
                                          0.5,
                                          True,
                                          layerwise=layerwise,
                                          fuse_scaling=fuse_scaling) as aem:
            model(inp)
    scale_factor_regions = aem.scale_factors
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)

    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    assert any([shape != () for shape in shape_scale_regions])


@requires_pt_ge('2.4')
@pytest_cases.parametrize('partial_had', [True, False])
def test_models(rotation_fixtures, partial_had):

    in_shape = IN_SIZE_LINEAR

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


# This method is almost the same as brevitas.graph.equalize.random_orthogonal_matrix, except for the
# possibility of passing a generator, that enables controlling the random matrices that are generated
# Adapted from https://github.com/facebookresearch/SpinQuant/blob/main/eval_utils/rotation_utils.py#L26
# This functions needs to be patches to enable passing the generator and ensuring that the orthogonal
# matrices generated are the same.
def _random_orthogonal_matrix(size, generator):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    Args:
    size (int): The size of the matrix (size x size).
    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64, generator=generator)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0).float()
    return q


# Auxiliar method to convert a dictionary of sources/sinks into a valid region
def _instantiate_region(region_dict, model) -> Region:
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
    return Region(
        srcs=sorted_srcs, sinks=sorted_sinks, acts=sorted_acts, name_to_module=model._modules)


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
def test_apply_rotate(rotation_model, mask, full_rotation_method, device, fuse_rotations, use_fx):
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
        graph_model, _ = torch._dynamo.export(model)(sample_inputs)
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
    regions_unfused = list(
        map(lambda x: _instantiate_region(x, rotated_model_unfused), regions_dicts))
    if full_rotation_method == 'had':
        # _apply_ort_device is patched to ensure that the hadamard matrices in hadamard.pt are used, instead of
        # the random ones generated by random_hadamard_matrices
        with patch('brevitas.graph.equalize._apply_ort_device',
                   lambda tensor,
                   had_K,
                   K: _apply_had_device(
                       tensor, get_hadK(had_K.shape[0])[0], get_hadK(had_K.shape[0])[1])):
            rewriters = _apply_rotate(
                rotated_model_unfused,
                regions_unfused,
                full_rotation_method=full_rotation_method,
                fuse_rotations=False)
    elif full_rotation_method == 'ort':
        with patch('brevitas.graph.equalize.random_orthogonal_matrix',
                   partial(_random_orthogonal_matrix, generator=generator)):
            rewriters = _apply_rotate(
                rotated_model_unfused,
                regions_unfused,
                full_rotation_method=full_rotation_method,
                fuse_rotations=False)
    # Register parametrizations after calling _apply_rotate, as these are not inmediately registered since they alter the structure of the
    # model, thus potentially causing a crash if the model is offloaded
    for r in rewriters:
        if isinstance(r, ModuleInstanceRegisterParametrization):
            rotated_model_unfused = r.apply(rotated_model_unfused)
    # Apply rotations on the model with fused rotations
    with patch('brevitas.graph.equalize.random_orthogonal_matrix',
               partial(_random_orthogonal_matrix, generator=generator_clone)):
        regions_fused = list(
            map(lambda x: _instantiate_region(x, rotated_model_fused), regions_dicts))
        _apply_rotate(
            rotated_model_fused,
            regions_fused,
            full_rotation_method=full_rotation_method,
            fuse_rotations=True)

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
