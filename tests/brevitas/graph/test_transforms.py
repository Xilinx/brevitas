# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import operator

from packaging import version
import pytest
import torch
from torch import nn
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.graph import AvgPoolToQuantDepthwiseConv
from brevitas.graph import DuplicateSharedStatelessModule
from brevitas.graph import FnToModule
from brevitas.graph import MeanMethodToAdaptiveAvgPool2d
from brevitas.graph import MergeBatchNorm
from brevitas.graph import MethodToModule
from brevitas.graph.base import ModuleToModuleByInstance
from brevitas.nn import QuantConv1d
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantConv3d

SEED = 123456
INPUT_SIZE = (1, 3, 224, 224)
ATOL = 1e-3

from brevitas import config

config.IGNORE_MISSING_KEYS = True

MODELS = ['mobilenet_v2', 'resnet18', 'mnasnet0_5']


@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("model_name", MODELS)
def test_rewriter_merge_bn(model_name: str, pretrained: bool):
    model = getattr(models, model_name)(pretrained=pretrained)
    model = model.train(False)
    torch.manual_seed(SEED)
    inp = torch.randn(INPUT_SIZE)
    with torch.no_grad():
        graph_model = symbolic_trace(model)
        graph_model.load_state_dict(model.state_dict())
        graph_model.train(False)
        torch.manual_seed(SEED)
        out_gen_model = graph_model(inp)
        graph_model = MergeBatchNorm().apply(graph_model)
        torch.manual_seed(SEED)
        out_gen_model_bn_fused = graph_model(inp)
        is_close = out_gen_model_bn_fused.isclose(out_gen_model, atol=ATOL).all().item()
        for m in graph_model.modules():
            assert not isinstance(m, nn.BatchNorm2d)
        assert is_close


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_conv_merge_bn(dims):

    class TestModel(nn.Module):

        def __init__(self, dims):
            super(TestModel, self).__init__()
            layers = []

            if dims == 1:
                layers.append(nn.Conv1d(16, 33, 3, stride=2))
                layers.append(nn.BatchNorm1d(33))
            elif dims == 2:
                layers.append(nn.Conv2d(16, 33, 3, stride=2))
                layers.append(nn.BatchNorm2d(33))
            else:
                layers.append(nn.Conv3d(16, 33, 3, stride=2))
                layers.append(nn.BatchNorm3d(33))

            layers.append(nn.ReLU())

            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    model = TestModel(dims)
    graph = symbolic_trace(model)
    graph = MergeBatchNorm().apply(graph)

    for m in graph.modules():
        if dims == 1:
            assert not isinstance(m, nn.BatchNorm1d)
        elif dims == 2:
            assert not isinstance(m, nn.BatchNorm2d)
        else:
            assert not isinstance(m, nn.BatchNorm3d)


@pytest.mark.parametrize("dims", [1, 2, 3])
def test_avg_pool_to_quant_conv(dims):

    class TestModel(nn.Module):

        def __init__(self, dims):
            super(TestModel, self).__init__()

            if dims == 1:
                self.net = nn.Sequential(nn.AvgPool1d(3, stride=2), nn.ReLU())
            elif dims == 2:
                self.net = nn.Sequential(nn.AvgPool2d(3, stride=2), nn.ReLU())
            else:
                self.net = nn.Sequential(nn.AvgPool3d(3, stride=2), nn.ReLU())

        def forward(self, x):
            return self.net(x)

    model = TestModel(dims)

    args = None
    if dims == 1:
        args = torch.randn(20, 16, 10)
    elif dims == 2:
        args = torch.randn(20, 16, 10, 50)
    else:
        args = torch.randn(20, 16, 10, 50, 100)

    graph = symbolic_trace(model)
    graph = AvgPoolToQuantDepthwiseConv().apply(graph, args)

    has_quant_conv = False
    for m in graph.modules():
        if isinstance(m, (QuantConv1d, QuantConv2d, QuantConv3d)):
            has_quant_conv = True

        assert not isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d))

    assert has_quant_conv


def test_rewriter_duplicate_shared_relu():

    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.act(x))

    model = TestModel()
    graph_model = symbolic_trace(model)
    deduplicate_graph_model = DuplicateSharedStatelessModule().apply(graph_model)
    assert hasattr(deduplicate_graph_model, 'act_1')
    assert deduplicate_graph_model.act is not deduplicate_graph_model.act_1


def test_rewriter_duplicate_nested_shared_relu():

    class TestSubModel(nn.Module):

        def __init__(self):
            super(TestSubModel, self).__init__()
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.act(x))

    class TestModel(nn.Module):

        def __init__(self):
            super(TestModel, self).__init__()
            self.sub_model = TestSubModel()

        def forward(self, x):
            return self.sub_model(x)

    model = TestModel()
    graph_model = symbolic_trace(model)
    deduplicate_graph_model = DuplicateSharedStatelessModule().apply(graph_model)
    assert hasattr(deduplicate_graph_model.sub_model, 'act_1')
    assert deduplicate_graph_model.sub_model.act is not deduplicate_graph_model.sub_model.act_1


def test_rewriter_add_fn_to_module():

    class TestModel(nn.Module):

        def forward(self, x):
            return torch.add(x, x)

    model = TestModel()
    graph_model = symbolic_trace(model)
    graph_model = FnToModule(torch.add, TestModel).apply(graph_model)
    # Due to changes in fx after 1.8
    attr_check = getattr(graph_model, 'add_1', None) or getattr(graph_model, 'add', None)
    assert isinstance(attr_check, TestModel)


def test_rewriter_max_pool_to_module():

    class TestModel(nn.Module):

        def forward(self, x):
            return torch.max_pool2d(x, 2, stride=2, padding=1, dilation=1)

    model = TestModel()
    graph_model = symbolic_trace(model)
    graph_model = FnToModule(torch.max_pool2d, nn.MaxPool2d).apply(graph_model)
    inp = torch.randn(2, 10, 10)
    # Due to changes in fx after 1.8
    attr_check = getattr(graph_model, 'max_pool2d_1', None) or getattr(
        graph_model, 'max_pool2d', None)
    assert isinstance(attr_check, nn.MaxPool2d)
    assert (model(inp) == graph_model(inp)).all().item()


def test_rewriter_add_method_to_module():

    class AddModule(nn.Module):

        def forward(self, x, y):
            return torch.add(x, y)

    class TestModel(nn.Module):

        def forward(self, x):
            return x.add(x)

    model = TestModel()
    graph_model = symbolic_trace(model)
    graph_model = MethodToModule('add', AddModule).apply(graph_model)
    inp = torch.randn(2, 10, 10)
    # Due to changes in fx after 1.8
    attr_check = getattr(graph_model, 'add_1', None) or getattr(graph_model, 'add', None)
    assert isinstance(attr_check, AddModule)
    assert (model(inp) == graph_model(inp)).all().item()


def test_rewriter_add_magic_to_module():

    class AddModule(nn.Module):

        def forward(self, x, y):
            return torch.add(x, y)

    class TestModel(nn.Module):

        def forward(self, x):
            return x + x

    model = TestModel()
    graph_model = symbolic_trace(model)
    graph_model = FnToModule(operator.add, AddModule).apply(graph_model)
    inp = torch.randn(2, 10, 10)
    # Due to changes in fx after 1.8
    attr_check = getattr(graph_model, 'add_1', None) or getattr(graph_model, 'add', None)
    assert isinstance(attr_check, AddModule)
    assert (model(inp) == graph_model(inp)).all().item()


def test_rewriter_mean_to_module():

    class TestModel(nn.Module):

        def forward(self, x):
            return x.mean((2, 3))

    model = TestModel()
    graph_model = symbolic_trace(model)
    graph_model = MeanMethodToAdaptiveAvgPool2d().apply(graph_model)
    inp = torch.randn(2, 3, 10, 10)
    # Due to changes in fx after 1.8
    attr_check = getattr(graph_model, 'mean_1', None) or getattr(graph_model, 'mean', None)
    assert isinstance(attr_check, nn.AdaptiveAvgPool2d)
    assert (model(inp) == graph_model(inp)).all().item()


def test_lambda_kwargs():

    class TestModel(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3)

        def forward(self, x):
            return self.conv(x)

    model = TestModel()
    assert model.conv.stride == (1, 1)

    kwargs = {'stride': lambda module, name: 2 if module.in_channels == 3 else 1}
    model = ModuleToModuleByInstance(model.conv, nn.Conv2d, **kwargs).apply(model)
    assert model.conv.stride == (2, 2)
