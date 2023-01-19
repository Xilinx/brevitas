# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import os

from packaging import version
import pytest
from pytest_cases import fixture
from pytest_cases import parametrize
import torch
import torchvision.models as modelzoo

from brevitas import torch_version
from brevitas.export import export_onnx_qcdq
from brevitas.export import export_torch_qcdq
from tests.marker import requires_pt_ge

BATCH = 1
HEIGHT, WIDTH = 224, 224
IN_CH = 3
MODEL_LIST = ['mobilenet_v2', 'resnet50', 'resnet18', 'mnasnet0_5', 'alexnet', 'googlenet', 'vgg11', 'densenet121', 'deeplabv3_resnet50', 'fcn_resnet50', 'regnet_x_400mf','squeezenet1_0', 'inception_v3']

class NoDictModel(torch.nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return out['out']

@fixture
@parametrize('model_name', MODEL_LIST)
def torchvision_model(model_name):
    from brevitas.graph.target.flexml import preprocess_flexml
    from brevitas.graph.target.flexml import quantize_flexml

    inp =  torch.randn(BATCH, IN_CH, HEIGHT, WIDTH)

    if torch_version <= version.parse('1.9.1') and model_name == 'regnet_x_400mf':
        return None

    # Deeplab and fcn are in a different module, and they have a dict as output which is not suited for torchscript
    if model_name in ('deeplabv3_resnet50', 'fcn_resnet50'):
        model_fn = getattr(modelzoo.segmentation, model_name)
        model = NoDictModel(model_fn(pretrained=False, aux_loss=False))
    else:
        model_fn = getattr(modelzoo, model_name)
        model = model_fn(pretrained=False)

    model.eval()
    model = preprocess_flexml(model, inp)
    model = quantize_flexml(model)
    return model


@requires_pt_ge('1.8.1')
def test_torchvision_graph_quantization_flexml_qcdq_onnx(torchvision_model, request):
    if torchvision_model is None:
        pytest.skip('Model not instantiated')
    test_id = request.node.callspec.id
    inp =  torch.randn(BATCH, IN_CH, HEIGHT, WIDTH)
    export_onnx_qcdq(
        torchvision_model, args=inp, export_path=test_id +'model_onnx_qcdq.onnx')

@requires_pt_ge('1.9.1')
def test_torchvision_graph_quantization_flexml_qcdq_torch(torchvision_model, request):
    if torchvision_model is None:
        pytest.skip('Model not instantiated')
    test_id = request.node.callspec.id
    inp =  torch.randn(BATCH, IN_CH, HEIGHT, WIDTH)
    export_torch_qcdq(
        torchvision_model, args=inp, export_path=test_id + 'model_torch_qcdq.pt')
