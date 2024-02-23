# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torch import Tensor
from torch.nn import Module
from torchvision.models import alexnet
from torchvision.models import densenet121
from torchvision.models import mnasnet0_5
from torchvision.models import mobilenet_v2
from torchvision.models import resnet18
from torchvision.models import shufflenet_v2_x0_5
from torchvision.models import squeezenet1_0

from brevitas.fx import brevitas_symbolic_trace
from brevitas.fx import value_trace
from brevitas.quant_tensor import QuantTensor

SEED = 123456
INPUT_SIZE = (2, 3, 224, 224)
INCEPTION_INPUT_SIZE = (2, 3, 299, 299)

TV_MODELS = [
    resnet18, mobilenet_v2, alexnet, squeezenet1_0, shufflenet_v2_x0_5, mnasnet0_5, densenet121]


@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("model_impl", TV_MODELS)
def test_value_tracer(model_impl, train: bool, pretrained: bool):
    model = model_impl(pretrained=pretrained)
    model = model.train(train)
    torch.manual_seed(SEED)
    inp = torch.randn(INPUT_SIZE)
    # Use a different input for tracing
    # to test for baked in values
    trace_inp = torch.randn(INPUT_SIZE)
    torch.manual_seed(SEED)
    with torch.no_grad():
        graph_model = value_trace(model, value_args={'x': trace_inp})
        torch.manual_seed(SEED)
        out = model(inp)
        torch.manual_seed(SEED)
        graph_out = graph_model(inp)
        if isinstance(out, (tuple, list)):
            assert isinstance(graph_out, (tuple, list))
            for reference, o in zip(out, graph_out):
                assert o.isclose(reference).all().item()
        else:
            assert graph_out.isclose(out).all().item()


@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("model_impl", TV_MODELS)
def test_brevitas_symbolic_tracer(model_impl, train: bool, pretrained: bool):
    model = model_impl(pretrained=pretrained)
    model = model.train(train)
    torch.manual_seed(SEED)
    inp = torch.randn(INPUT_SIZE)
    torch.manual_seed(SEED)
    with torch.no_grad():
        graph_model = brevitas_symbolic_trace(model)
        torch.manual_seed(SEED)
        out = model(inp)
        torch.manual_seed(SEED)
        graph_out = graph_model(inp)
        if isinstance(out, (tuple, list)):
            assert isinstance(graph_out, (tuple, list))
            for reference, o in zip(out, graph_out):
                assert o.isclose(reference).all().item()
        else:
            assert graph_out.isclose(out).all().item()


class UnpackShape(Module):

    def forward(self, x):
        size = x.size()
        batchsize, num_channels, height, width = size
        return x


class ReshapeModule(Module):

    def forward(self, x):
        groups = 1
        batchsize, num_channels, height, width = x.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        return x


class CatChunkUnrolledModule(Module):

    def forward(self, x: Tensor):
        x1, x2 = x.chunk(2, dim=1)
        x = torch.cat([x1, x2], dim=1)
        return x


class CatChunkRolledModule(Module):

    def forward(self, x: Tensor):
        x = x.chunk(2, dim=1)
        x = torch.cat(x, dim=1)
        return x


class CatChunkUnpackModule(Module):

    def forward(self, x: Tensor):
        x = x.chunk(2, dim=1)
        x = torch.cat(tuple(v for v in x), dim=1)
        return x


class InPlaceTorchAddModule(Module):

    def forward(self, x: Tensor):
        x.add_(x)
        return x


class InPlacePythonAddModule(Module):

    def forward(self, x: Tensor):
        x += x
        return x


class PythonAddModule(Module):

    def forward(self, x: Tensor):
        x = x + x
        return x


class TorchAddModule(Module):

    def forward(self, x: Tensor):
        x = torch.add(x, x)
        return x


class TorchNoneModule(Module):

    def forward(self, x: Tensor):
        y = x.clone()
        y = None
        if y is None:
            x = torch.add(x, x)
        return x


class TorchNone2Module(Module):

    def fn(self, x, y=None):
        if y is None:
            return x + 10
        else:
            return x

    def forward(self, x: Tensor):
        self.fn(x, None)
        return x


class TorchCondModule(Module):

    def forward(self, x: Tensor):
        x.fill_(1.)
        if (x > 0).all():
            x = torch.add(x, x)
        return x


class TorchIsInstanceModule(Module):

    def forward(self, x: Tensor):
        print(x.__class__)
        if isinstance(x, torch.Tensor):
            x = torch.add(x, x)
        return x


class QuantTensorInputModule(torch.nn.Module):

    def forward(self, x):
        if isinstance(x, QuantTensor):
            return x + x
        else:
            return x


MODULES = [
    UnpackShape,
    ReshapeModule,
    InPlaceTorchAddModule,
    InPlacePythonAddModule,
    TorchAddModule,
    PythonAddModule,
    CatChunkUnrolledModule,
    CatChunkRolledModule,
    CatChunkUnpackModule,
    TorchNoneModule,
    TorchNone2Module,
    TorchIsInstanceModule,
    TorchCondModule]

QUANT_TENSOR_MODULES = [QuantTensorInputModule]


@pytest.mark.parametrize('module', MODULES)
def test_module(module):
    mod = module()
    x = torch.randn(INPUT_SIZE)
    x_trace = torch.randn(INPUT_SIZE)
    with torch.no_grad():
        out = mod(x.clone())
        graph_model = value_trace(mod, value_args={'x': x_trace})
        graph_out = graph_model(x.clone())
        if isinstance(out, (tuple, list)):
            assert isinstance(graph_out, (tuple, list))
            for reference, o in zip(out, graph_out):
                assert o.isclose(reference).all().item()
        else:
            assert graph_out.isclose(out).all().item()


@pytest.mark.parametrize('module', QUANT_TENSOR_MODULES)
def test_quant_module(module):
    mod = module()
    x = torch.randn(INPUT_SIZE)
    x_trace = torch.randn(INPUT_SIZE)
    with torch.no_grad():
        out = mod(x)
        graph_model = value_trace(mod, value_args={'x': x_trace})
        graph_out = graph_model(x)
        assert graph_out.isclose(out).all().item()
