# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import copy

import torch
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _is_supported_module
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
        model, regions, merge_bias=True, bias_shrinkage='vaiq', scale_computation_type='maxabs')
    out = model(inp)

    # Check that equalization is not introducing FP variations
    assert torch.allclose(expected_out, out, atol=ATOL)

    regions = sorted(regions, key=lambda region: region.srcs[0])
    resnet_18_regions = sorted(RESNET_18_REGIONS, key=lambda region: region[0][0])
    equalized_layers = set()
    for r in resnet_18_regions:
        equalized_layers.update(r[0])
        equalized_layers.update(r[1])

    # Check that we found all the expected regions
    for region, expected_region in zip(regions, resnet_18_regions):
        sources_check = set(region.srcs) == set(expected_region[0])
        sinks_check = set(region.sinks) == set(expected_region[1])
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

    expected_out = model(inp)

    regions = _extract_regions(model)
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
    count = 0
    for r in regions:
        srcs.update(list(r.srcs))
        sinks.update(list(r.sinks))

    for n in model.graph.nodes:
        if _is_supported_module(model, n):
            count += 1
    src_coverage = len(srcs) / count
    sink_coverage = len(sinks) / count
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
    expected_out = model(inp)
    model = symbolic_trace(model)
    regions = _extract_regions(model)
    scale_factor_regions = equalize_test(
        model,
        regions,
        merge_bias=merge_bias,
        bias_shrinkage='vaiq',
        scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    assert all([shape != () for shape in shape_scale_regions])
