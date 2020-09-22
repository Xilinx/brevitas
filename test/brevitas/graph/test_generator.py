import pytest

import torch
from torchvision import models

from brevitas.graph.tracer import Tracer
from brevitas.graph.generator import ModuleGenerator
from brevitas_examples.bnn_pynq import lfc_1w1a

SEED = 123456
IMAGENET_SIZE = (1, 3, 224, 224)
INCEPTION_IMAGENET_SIZE = (2, 3, 299, 299)
LARGER_IMAGE_SIZE = (2, 3, 340, 340)
MNIST_SIZE = (1, 1, 28, 28)

MODEL_NAMES = [
    'shufflenet_v2_x0_5',
    'googlenet',
    'inception_v3',
    'alexnet',
    'squeezenet1_0',
    'mnasnet0_5',
    'densenet121',
    'vgg11_bn',
    'resnet18',
    'mobilenet_v2']

@pytest.mark.parametrize("pretrained", [True, False])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_generator_torchvision(model_name: str, train: bool, pretrained: bool):
    if model_name == 'googlenet' and not pretrained and not train \
            or model_name == 'inception_v3' and not pretrained and not train \
            or model_name == 'inception_v3' and pretrained and not train:
        pytest.skip("Unsupport train/pretrained combo.")
    model = getattr(models, model_name)(pretrained=pretrained)
    model = model.train(train)
    torch.manual_seed(SEED)
    input = torch.randn(INCEPTION_IMAGENET_SIZE if model_name == 'inception_v3' else IMAGENET_SIZE)
    torch.manual_seed(SEED)
    trace = Tracer(input).trace_model(model)
    identity_model = ModuleGenerator().gen_model(trace)
    identity_model.load_state_dict(model.state_dict())

    model.train(train)
    identity_model.train(train)

    # Define a new input tensor to check we are not just memorizing stuff
    torch.manual_seed(SEED + 1)
    inp2 = torch.randn(INCEPTION_IMAGENET_SIZE if model_name == 'inception_v3' else IMAGENET_SIZE)
    torch.manual_seed(SEED)
    out = model(inp2)
    torch.manual_seed(SEED)
    q_out = identity_model(inp2)
    if isinstance(q_out, (tuple, list)):
        assert isinstance(out, (tuple, list))
        for reference, o in zip(out, q_out):
            assert o.isclose(reference).all().item()
    else:
        assert q_out.isclose(out).all().item()


# TODO fix tracing and generating a graph containing ScriptModules
@pytest.mark.skip(reason="This is still broken")
def test_generator_bnn_pynq(train=False):
    model = lfc_1w1a(pretrained=False)
    model = model.train(train)

    torch.manual_seed(SEED)
    input = torch.randn(MNIST_SIZE)

    model(input)
    trace = Tracer(input).trace_model(model, wrap_torchscript=True)
    quant_model = ModuleGenerator().gen_model(trace)
    quant_model.load_state_dict(model.state_dict())
    quant_model.train(train)
    for inst in quant_model.schedule:
        print(inst)
    torch.manual_seed(SEED)
    out = model(input)
    torch.manual_seed(SEED)
    q_out = quant_model(input)

    assert out.isclose(q_out).all().item()