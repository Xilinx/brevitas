# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import torch
from torchvision import models

from brevitas import torch_version
from brevitas.fx import symbolic_trace
from brevitas.graph import EqualizeGraph
from brevitas.graph.equalize import _is_supported_module

from .equalization_fixtures import *

SEED = 123456
IN_SIZE = (1, 3, 224, 224)
ATOL = 1e-3

@pytest_cases.parametrize("model_dict", [(model_name, coverage) for model_name, coverage in MODELS.items()], ids=[ model_name for model_name, _ in MODELS.items()])
def test_equalization_torchvision_models(model_dict: dict):
    model, coverage = model_dict

    if model == 'googlenet' and torch_version == version.parse('1.8.1'):
        pytest.skip('Skip because of PyTorch error = AttributeError: \'function\' object has no attribute \'GoogLeNetOutputs\' ')

    try:
        model = getattr(models, model)(pretrained=True, transform_input=False)
    except TypeError:
        model = getattr(models, model)(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE)
    model.eval()
    expected_out = model(inp)

    model = symbolic_trace(model)
    model, regions = EqualizeGraph(3, return_regions=True).apply(model)

    out = model(inp)
    srcs = set()
    sinks = set()
    count= 0
    for r in regions:
        srcs.update(list(r[0]))
        sinks.update(list(r[1]))

    for n in model.graph.nodes:
        if _is_supported_module(model, n):
            count += 1
    src_coverage = len(srcs)/count
    sink_coverage = len(sinks)/count
    assert src_coverage >= coverage[0]
    assert sink_coverage >= coverage[1]
    assert torch.allclose(expected_out, out, atol=ATOL)

def test_models(toy_model):
    model = toy_model()
    inp = torch.randn(IN_SIZE)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)
    model, regions = EqualizeGraph(3, return_regions=True).apply(model)

    out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
