import pytest
from packaging import version

import torch
from torchvision import models

from brevitas.export import export_dpuv2_onnx
from brevitas.graph.quantizer import quantize, BatchNormHandling
from brevitas.quant.fixed_point import *
from brevitas import config

from tests.marker import requires_pt_ge

config.IGNORE_MISSING_KEYS = True


MODELS = [
    'resnet18',
    'mobilenet_v2',
    'mnasnet0_5'
]

IN_SIZE = (1, 3, 224, 224)


@pytest.mark.parametrize("model_name", MODELS)
def test_rewriter_export(model_name: str):
    model = getattr(models, model_name)(pretrained=True)
    model = model.train(True)
    input = torch.randn(IN_SIZE)
    gen_model = quantize(
        model, input,
        weight_quant=Int8WeightPerTensorFixedPoint,
        input_quant=Int8ActPerTensorFixedPoint,
        output_quant=Int8ActPerTensorFixedPoint,
        bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
        bn_handling=BatchNormHandling.MERGE_AND_QUANTIZE)
    out_file = f'{model_name}.onnx'
    export_dpuv2_onnx(gen_model, input_t=input, export_path=out_file)
