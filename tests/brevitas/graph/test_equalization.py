# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import torch
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.graph.equalize import _batch_norm
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _is_supported_module
from brevitas.graph.equalize import activation_equalization_mode
from brevitas.graph.standardize import DuplicateSharedStatelessModule
from brevitas.graph.standardize import TorchFunctionalToModule
from brevitas.graph.utils import get_module

from .equalization_fixtures import *


def test_resnet18_equalization():
    model = models.resnet18(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()
    model = symbolic_trace(model)
    expected_out = model(inp)

    model_orig = copy.deepcopy(model)
    regions = _extract_regions(model)
    _ = equalize_test(
        regions, merge_bias=True, bias_shrinkage='vaiq', scale_computation_type='maxabs')
    out = model(inp)

    # Check that equalization is not introducing FP variations
    assert torch.allclose(expected_out, out, atol=ATOL)

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

    regions = _extract_regions(model)
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
    regions = _extract_regions(model)
    scale_factor_regions = equalize_test(
        regions, merge_bias=merge_bias, bias_shrinkage='vaiq', scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    with torch.no_grad():
        out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
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
    if 'srcsinkconflict_mode' not in test_id:
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
