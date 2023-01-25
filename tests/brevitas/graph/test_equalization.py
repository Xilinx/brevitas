# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import operator
from statistics import mode
from tokenize import group
from turtle import forward

from packaging import version
import pytest
import pytest_cases
from pytest_cases import fixture_union
import torch
from torch import nn
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.fx import value_trace
from brevitas.graph import AdaptiveAvgPoolToAvgPool
from brevitas.graph import CollapseConsecutiveConcats
from brevitas.graph import DuplicateSharedStatelessModule
from brevitas.graph import EqualizeGraph
from brevitas.graph import MeanMethodToAdaptiveAvgPool2d
from brevitas.graph import MergeBatchNorm
from brevitas.graph import ModuleToModuleByClass
from brevitas.graph import MoveSplitBatchNormBeforeCat
from brevitas.graph import TorchFunctionalToModule
from brevitas.graph.equalize import _is_supported_module

from .equalization_fixtures import *

SEED = 123456
IN_SIZE = (16,3,224,224)
ATOL = 1e-3


@pytest.mark.parametrize("model_name", MODELS)
def test_rewriter_merge_bn(model_name: str):
    try:
        model = getattr(models, model_name)(pretrained=True, transform_input=False)
    except:
        model = getattr(models, model_name)(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE)
    model.eval()
    expected_out = model(inp)

    model = value_trace(model)
    model = TorchFunctionalToModule().apply(model)
    model = DuplicateSharedStatelessModule().apply(model)
    model = MeanMethodToAdaptiveAvgPool2d().apply(model)
    model = AdaptiveAvgPoolToAvgPool().apply(model, inp)
    model = CollapseConsecutiveConcats().apply(model)
    model = MoveSplitBatchNormBeforeCat().apply(model)
    model = MergeBatchNorm().apply(model)
    model, regions = EqualizeGraph(1).apply(model)


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

    print(f"Source coverage {len(srcs)/count}")
    print(f"Sink coverage {len(sinks)/count}")
    assert torch.allclose(expected_out, out, atol=ATOL)





def test_models(all_models):
    model = all_models()
    inp = torch.randn(IN_SIZE)

    model.eval()
    expected_out = model(inp)
    model = value_trace(model)
    model, regions = EqualizeGraph(3).apply(model)

    out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
