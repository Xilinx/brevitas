import pytest
import torch
from torchvision import models

from brevitas.onnx import export_dpuv1_onnx
from brevitas.graph.quantizer import quantize, BatchNormHandling
from brevitas import config
config.IGNORE_MISSING_KEYS = True

MODELS = [
    'mnasnet0_5',
    'resnet18',
    'mobilenet_v2',
]

IN_SIZE = (1, 3, 224, 224)


@pytest.mark.parametrize("model_name", MODELS)
def test_rewriter_export(model_name: str):
    model = getattr(models, model_name)(pretrained=True)
    model = model.train(True)
    input = torch.randn(IN_SIZE)
    gen_model = quantize(
        model, input,
        power_of_two_scaling=True,
        per_output_channel=False,
        weight_bit_width=8,
        act_bit_width=8,
        bias_bit_width=8,
        bn_handling=BatchNormHandling.MERGE_AND_QUANTIZE)
    out_file = f'{model_name}.onnx'
    export_dpuv1_onnx(gen_model, input_shape=IN_SIZE, input_t=input, export_path=out_file)