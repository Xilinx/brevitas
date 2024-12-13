# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy
import itertools
from typing import List, Tuple
from unittest.mock import patch

import pytest
import torch
import torch.nn.utils.parametrize as parametrize
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.graph.equalize import _apply_ort_device
from brevitas.graph.equalize import _batch_norm
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _get_input_axis
from brevitas.graph.equalize import _get_output_axis
from brevitas.graph.equalize import _is_supported_module
from brevitas.graph.equalize import _supported_layers
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import MergeLnAffine
from brevitas.graph.equalize import random_orthogonal_matrix
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.graph.utils import get_module
from brevitas.nn.equalized_layer import RotationBiasParametrization
from brevitas.nn.equalized_layer import RotationWeightParametrization
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
        regions, merge_bias=True, bias_shrinkage='vaiq', scale_computation_type='maxabs')
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
        regions, merge_bias=merge_bias, bias_shrinkage='vaiq', scale_computation_type='maxabs')
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
        regions, merge_bias=merge_bias, bias_shrinkage='vaiq', scale_computation_type='maxabs')
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
def test_act_equalization_models(toy_model, layerwise, request):
    test_id = request.node.callspec.id

    if 'mha' in test_id:
        in_shape = IN_SIZE_LINEAR
    else:
        in_shape = IN_SIZE_CONV

    model_class = toy_model
    model = model_class()
    inp = torch.randn(in_shape)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)
    with torch.no_grad():
        with activation_equalization_mode(model, 0.5, True, layerwise=layerwise) as aem:
            regions = aem.graph_act_eq.regions
            model(inp)
    scale_factor_regions = aem.scale_factors
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL)

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
def test_act_equalization_torchvision_models(model_dict: dict, layerwise: bool):
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
        with activation_equalization_mode(model, 0.5, True, layerwise=layerwise) as aem:
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
    penultimate_weight = model.linear_1.weight.data
    last_weight = model.linear_2.weight.data
    with torch.no_grad():
        expected_out = model(inp)

    model = symbolic_trace(model)
    merge = MergeLnAffine()
    model = merge.apply(model)
    eq = GraphRotationEqualization(orphan_sink=partial_had)
    model = eq.apply(model)

    with torch.no_grad():
        out = model(inp)

    penultimate_weight_new = model.linear_1.weight.data

    # Invariance of the output
    assert torch.allclose(out, expected_out, atol=ATOL)
    # Rotate weights must be different
    assert not torch.allclose(penultimate_weight, penultimate_weight_new)
    # Merging affine parameters of RMS
    assert torch.allclose(model.rms.weight.data, torch.ones_like(model.rms.weight.data))
    if partial_had:
        last_weight_new = model.linear_2.layer.weight.data
        assert not torch.allclose(last_weight, last_weight_new)


def _rotate_input_output(is_source: bool, is_sink: bool, is_orphan: bool) -> Tuple[bool, bool]:
    # Verify that only one flag is enabled simultaneously
    assert sum([is_source, is_sink, is_orphan]) <= 1, "Only one flag can be enabled."

    rotate_input, rotate_output = False, False
    if is_source:
        rotate_output = True
    if is_sink or is_orphan:
        rotate_input = True

    return rotate_input, rotate_output


def _compute_rotated_ouptut_from_matrices(
        module: nn.Module, inp: torch.Tensor, rot_mat_input: torch.Tensor,
        rot_mat_output: torch.Tensor):
    # If the node is a sink, the input is multiplied by the inverse of the rotation matrix x <- xQ^{-1}
    inp = inp @ rot_mat_input.t()
    # If the node is a source, the output is multiplied by the rotation matrix o <- oQ
    out = module(inp) @ rot_mat_output
    # Return rotated output
    return out


# RotationParametrizations can only have one type flag enabled simultaneously (is_source, is_sink, is_orphan).
# Moreover, orphan rotations need to be the outermost rotation, as this cancels out when rotating the input.
def _generate_rotation_flags(N: int) -> List[bool]:
    return [
        rotation_flags for rotation_flags in itertools.product([False, True], repeat=3 * N) if (
            all([sum(rotation_flags[i * 3:(i + 1) * 3]) <= 1
                 for i in range(N)]) and all([not rotation_flags[i * 3 + 2] for i in range(N - 1)]))
    ]


@requires_pt_ge('2.4')
@pytest_cases.parametrize('N', [1, 2, 3], ids=lambda x: f"N={x}")
def test_composition_unfused_rotations(N):
    torch.manual_seed(SEED)

    for rotation_flags in _generate_rotation_flags(N):

        in_features = IN_FEATURES_LINEAR
        module = nn.Linear(in_features=in_features, out_features=in_features)
        rot_module = copy.deepcopy(module)

        # Sample input to pass through the block
        sample_input = torch.rand((1, in_features),)
        # Composite rotation matrices
        rot_mat_input = torch.eye(in_features)
        rot_mat_output = torch.eye(in_features)

        for i in range(N):
            module_rotation_flags = rotation_flags[i * 3:(i + 1) * 3]
            is_source, is_sink, is_orphan = module_rotation_flags
            rotate_input, rotate_output = _rotate_input_output(is_source, is_sink, is_orphan)

            # Generate a random matrix
            rot_mat = random_orthogonal_matrix(in_features).to(dtype=torch.float32)

            # Aggregate rotation matrices
            if rotate_input:
                rot_mat_input = rot_mat_input @ rot_mat
            if rotate_output:
                rot_mat_output = rot_mat_output @ rot_mat

            # Compose rotation modules
            parametrize.register_parametrization(
                rot_module,
                "weight",
                RotationWeightParametrization(
                    rot_mat=rot_mat,
                    rot_func=_apply_ort_device,
                    input_axis=_get_input_axis(rot_module),
                    output_axis=_get_output_axis(rot_module),
                    is_source=is_source,
                    is_sink=is_sink,
                    is_orphan=is_orphan,
                ))
            parametrize.register_parametrization(
                rot_module,
                "bias",
                RotationBiasParametrization(
                    rot_mat=rot_mat,
                    rot_func=_apply_ort_device,
                    input_axis=_get_input_axis(rot_module),
                    output_axis=_get_output_axis(rot_module),
                    is_source=is_source,
                    is_sink=is_sink,
                    is_orphan=is_orphan,
                ))

        gt_output = _compute_rotated_ouptut_from_matrices(
            module, sample_input, rot_mat_input, rot_mat_output)
        rot_output = rot_module(sample_input)

        # Verify that the rotation operations were computed correctly
        assert torch.allclose(gt_output, rot_output, atol=ATOL)
