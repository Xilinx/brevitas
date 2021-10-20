import pytest
import torch
from torch import Tensor
from torch.nn import Module
from torchvision.models.resnet import resnet18
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.models.alexnet import alexnet
from torchvision.models.squeezenet import squeezenet1_0
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5
from torchvision.models.mnasnet import mnasnet0_5
from torchvision.models.densenet import densenet121

from brevitas.fx import value_trace, brevitas_symbolic_trace, brevitas_value_trace

SEED = 123456
INPUT_SIZE = (2, 3, 224, 224)
INCEPTION_INPUT_SIZE = (2, 3, 299, 299)


TV_MODELS = [
    resnet18,
    mobilenet_v2,
    alexnet,
    squeezenet1_0,
    shufflenet_v2_x0_5,
    mnasnet0_5,
    densenet121]


@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("model_impl", TV_MODELS)
def test_brevitas_value_tracer(model_impl, train: bool, pretrained: bool):
    model = model_impl(pretrained=pretrained)
    model = model.train(train)
    torch.manual_seed(SEED)
    inp = torch.randn(INPUT_SIZE)
    torch.manual_seed(SEED)
    with torch.no_grad():
        graph_model = brevitas_value_trace(model, concrete_args={'x': inp})
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


MODULES = [
    UnpackShape,
    ReshapeModule,
    InPlaceTorchAddModule,
    InPlacePythonAddModule,
    TorchAddModule,
    PythonAddModule,
    CatChunkUnrolledModule,
    CatChunkRolledModule]


@pytest.mark.parametrize('module', MODULES)
def test_module(module):
    mod = module()
    x = torch.randn(INPUT_SIZE)
    with torch.no_grad():
        out = mod(x.clone())
        graph_model = value_trace(mod, {'x': x.clone()})
        graph_out = graph_model(x.clone())
        if isinstance(out, (tuple, list)):
            assert isinstance(graph_out, (tuple, list))
            for reference, o in zip(out, graph_out):
                assert o.isclose(reference).all().item()
        else:
            assert graph_out.isclose(out).all().item()