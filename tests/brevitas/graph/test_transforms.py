import operator

from packaging import version
import pytest
import torch
from torch import nn
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.graph import DuplicateSharedStatelessModule
from brevitas.graph import FnToModule
from brevitas.graph import MethodToModule
from brevitas.graph import MeanMethodToAdaptiveAvgPool2d
from brevitas.graph import MergeBatchNorm

SEED = 123456
INPUT_SIZE = (1, 3, 224, 224)
ATOL = 1e-3

from brevitas import config
config.IGNORE_MISSING_KEYS = True

MODELS = [
    'mobilenet_v2',
    'resnet18',
    'mnasnet0_5'
]


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
    assert isinstance(graph_model.add_1, TestModel)


def test_rewriter_max_pool_to_module():

    class TestModel(nn.Module):

        def forward(self, x):
            return torch.max_pool2d(x, 2, stride=2, padding=1, dilation=1)

    model = TestModel()
    graph_model = symbolic_trace(model)
    graph_model = FnToModule(torch.max_pool2d, nn.MaxPool2d).apply(graph_model)
    inp = torch.randn(2, 10, 10)
    assert isinstance(graph_model.max_pool2d_1, nn.MaxPool2d)
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
    assert isinstance(graph_model.add_1, AddModule)
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
    assert isinstance(graph_model.add_1, AddModule)
    assert (model(inp) == graph_model(inp)).all().item()


def test_rewriter_mean_to_module():

    class TestModel(nn.Module):

        def forward(self, x):
            return x.mean((2, 3))

    model = TestModel()
    graph_model = symbolic_trace(model)
    graph_model = MeanMethodToAdaptiveAvgPool2d().apply(graph_model)
    inp = torch.randn(2, 3, 10, 10)
    assert isinstance(graph_model.mean_1, nn.AdaptiveAvgPool2d)
    assert (model(inp) == graph_model(inp)).all().item()
