import pytest
import torch
from torch import Tensor
from torch import nn
from torchvision import models

from brevitas.graph.tracer import Tracer
from brevitas.graph.generator import ModuleGenerator
from brevitas.graph.rewriter import MergeBatchNorm2d, DuplicateSharedStatelessModule

SEED = 123456
INPUT_SIZE = (1, 3, 224, 224)
ATOL = 1e-3

from brevitas import config
config.IGNORE_MISSING_KEYS = True

MODELS = [
    'mnasnet0_5',
    'mobilenet_v2',
    'vgg11_bn',
    'resnet18'
]

@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("model_name", MODELS)
def test_rewriter_merge_bn(model_name: str, pretrained: bool):
    model = getattr(models, model_name)(pretrained=pretrained)
    model = model.train(False)

    torch.manual_seed(SEED)
    input = torch.randn(INPUT_SIZE)
    trace = Tracer(input).trace_model(model)

    gen_model = ModuleGenerator().gen_model(trace)
    gen_model.load_state_dict(model.state_dict())
    gen_model.train(False)
    torch.manual_seed(SEED)
    out_gen_model = gen_model(input)
    gen_model_bn_fused = MergeBatchNorm2d().apply(gen_model)
    torch.manual_seed(SEED)
    out_gen_model_bn_fused = gen_model_bn_fused(input)
    is_close = out_gen_model_bn_fused.isclose(out_gen_model, atol=ATOL).all().item()
    for m in gen_model_bn_fused.modules():
        assert not isinstance(m, nn.BatchNorm2d)
    assert is_close


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("model_name", MODELS)
def test_rewriter_duplicate_shared_relu(model_name: str, train: bool):
    model = getattr(models, model_name)(pretrained=False)
    model = model.train(train)

    torch.manual_seed(SEED)
    input = torch.randn(INPUT_SIZE)
    trace = Tracer(input).trace_model(model)

    gen_model = ModuleGenerator().gen_model(trace)
    gen_model.load_state_dict(model.state_dict())
    gen_model.train(train)
    torch.manual_seed(SEED)
    out_gen_model = gen_model(input)
    gen_model_duplicate_relu = DuplicateSharedStatelessModule().apply(gen_model)
    torch.manual_seed(SEED)
    out_gen_model_duplicate_relu = gen_model_duplicate_relu(input)
    is_close = out_gen_model_duplicate_relu.isclose(out_gen_model, atol=ATOL).all().item()
    assert is_close