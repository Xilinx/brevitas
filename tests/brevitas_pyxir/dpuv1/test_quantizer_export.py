import pytest
from packaging import version

import torch
from torchvision import models

from brevitas.onnx import export_dpuv2_onnx
from brevitas.graph.quantizer import quantize, BatchNormHandling
from brevitas.quant.base import *
from brevitas import config
config.IGNORE_MISSING_KEYS = True

IS_ABOVE_110 = version.parse(torch.__version__) > version.parse("1.1.0")

MODELS = [
    'resnet18',
    'mobilenet_v2',
]

if IS_ABOVE_110:
    MODELS.append('mnasnet0_5')

IN_SIZE = (1, 3, 224, 224)


@pytest.mark.parametrize("model_name", MODELS)
def test_rewriter_export(model_name: str):
    model = getattr(models, model_name)(pretrained=True)
    model = model.train(True)
    input = torch.randn(IN_SIZE)
    bias_quant = IntQuant & MaxStatsScaling & PerTensorPoTScaling8bit
    weight_quant = NarrowIntQuant & MaxStatsScaling & PerTensorPoTScaling8bit
    io_quant = IntQuant & ParamFromRuntimePercentileScaling & PerTensorPoTScaling8bit
    gen_model = quantize(
        model, input,
        weight_quant=weight_quant,
        input_quant=io_quant,
        output_quant=io_quant,
        bias_quant=bias_quant,
        bn_handling=BatchNormHandling.MERGE_AND_QUANTIZE)
    out_file = f'{model_name}.onnx'
    export_dpuv2_onnx(gen_model, input_shape=IN_SIZE, input_t=input, export_path=out_file)