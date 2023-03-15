# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import torch
from torchvision import models

from brevitas import torch_version
from brevitas.fx import symbolic_trace
from brevitas.graph.equalize import _cross_layer_equalization
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.equalize import _is_supported_module

from .equalization_fixtures import *

SEED = 123456
ATOL = 1e-3


def equalize(model, regions, merge_bias, bias_shrinkage, scale_computation_type):
    name_to_module = {}
    name_set = {name for region in regions for module_set in region for name in module_set}
    scale_factors_regions = []
    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module
    for region in regions:
        scale_factors_region = _cross_layer_equalization([name_to_module[n] for n in region[0]], [name_to_module[n] for n in region[1]], merge_bias, bias_shrinkage, scale_computation_type)
        scale_factors_regions.append(scale_factors_region)
    return scale_factors_regions


@pytest_cases.parametrize("model_dict", [(model_name, coverage) for model_name, coverage in MODELS.items()], ids=[ model_name for model_name, _ in MODELS.items()])
@pytest.mark.parametrize("merge_bias", [True, False])
def test_equalization_torchvision_models(model_dict: dict, merge_bias: bool):
    model_name, coverage = model_dict

    if model_name == 'googlenet' and torch_version == version.parse('1.8.1'):
        pytest.skip('Skip because of PyTorch error = AttributeError: \'function\' object has no attribute \'GoogLeNetOutputs\' ')
    if 'vit' in model_name and torch_version < version.parse('1.13'):
        pytest.skip(f'ViT supported from torch version 1.13, current torch version is {torch_version}')

    try:
        model = getattr(models, model_name)(pretrained=True, transform_input=False)
    except TypeError:
        model = getattr(models, model_name)(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)
    model.eval()
    expected_out = model(inp)

    model = symbolic_trace(model)
    regions = _extract_regions(model)
    scale_factor_regions = equalize(model, regions, merge_bias=merge_bias, bias_shrinkage='vaiq', scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    srcs = set()
    sinks = set()
    count = 0
    for r in regions:
        srcs.update(list(r[0]))
        sinks.update(list(r[1]))

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
    print(sum([shape != () for shape in shape_scale_regions]))
    if 'alexnet' in model_name:
        # In AlexNet, we cannot equalize only through one region
        assert sum([shape == () for shape in shape_scale_regions]) == 1
    else:
        assert all([shape != () for shape in shape_scale_regions])


@pytest.mark.parametrize("merge_bias", [True, False])
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
    scale_factor_regions = equalize(model, regions, merge_bias=merge_bias, bias_shrinkage='vaiq', scale_computation_type='maxabs')
    shape_scale_regions = [scale.shape for scale in scale_factor_regions]

    out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
    # Check that at least one region performs "true" equalization
    # If all shapes are scalar, no equalization has been performed
    assert all([shape != () for shape in shape_scale_regions])
