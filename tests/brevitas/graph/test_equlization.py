# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import operator
from statistics import mode

from packaging import version
import pytest
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

SEED = 123456

MODELS = [
    'mobilenet_v2',
    'resnet18',
    'googlenet',
    'inception_v3',
    'alexnet'
]

@pytest.mark.parametrize("model_name", MODELS)
def test_rewriter_merge_bn(model_name: str):
    try:
        model = getattr(models, model_name)(pretrained=True, transform_input=False)
    except:
        model = getattr(models, model_name)(pretrained=True)
    model = model.train(False)
    torch.manual_seed(SEED)
    inp = torch.randn(16,3,224,224)

    expected_out = model(inp)

    model.eval()
    model = value_trace(model)
    model = TorchFunctionalToModule().apply(model)
    model = DuplicateSharedStatelessModule().apply(model)
    model = ModuleToModuleByClass(nn.ReLU6, nn.ReLU).apply(model)
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
    torch.allclose(expected_out, out)
