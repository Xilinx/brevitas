# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.quantize import preprocess_for_quantize
import brevitas.nn as qnn
from brevitas_examples.imagenet_classification.ptq.ptq_common import quantize_model

from .equalization_fixtures import *


class QuantConvModel(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = qnn.QuantConv2d(3, 16, 3)
        self.relu1 = qnn.QuantReLU(return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(16, 32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return x


def apply_gpfq(
        calib_loader,
        model,
        act_order,
        accumulator_bit_width: int = 32,
        a2q_layer_filter_fnc=lambda x: True):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        # use A2GPFQ if accumulator is less than 32 is specified
        with gpfq_mode(
                model,
                use_quant_activations=True,
                act_order=act_order,
                use_gpfa2q=accumulator_bit_width < 32,
                accumulator_bit_width=accumulator_bit_width,
                a2q_layer_filter_fnc=a2q_layer_filter_fnc,
        ) as gpfq:
            gpfq_model = gpfq.model
            for _ in range(gpfq.num_layers):
                for _, (images, _) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gpfq_model(images)
                gpfq.update()


def custom_layer_filter_fnc(layer: nn.Module) -> bool:
    if isinstance(layer, nn.Conv2d) and layer.in_channels == 3:
        return False
    return True


@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("acc_bit_width", [32, 24, 16])
def test_toymodels(toy_model, request, act_order: bool, acc_bit_width: int):
    model_name = request.node.callspec.id.split('-')[0]

    torch.manual_seed(SEED)

    model_class = toy_model
    model = model_class()

    # preprocess model for quantization, like merge BN etc.
    model = preprocess_for_quantize(model)
    # quantize model pretty basic
    model = quantize_model(
        model,
        backend='layerwise',
        weight_bit_width=8,
        act_bit_width=8,
        bias_bit_width=32,
        scale_factor_type='float_scale',
        weight_narrow_range=False,
        weight_param_method='stats',
        weight_quant_granularity='per_channel',
        weight_quant_type='sym',
        layerwise_first_last_bit_width=8,
        act_param_method='stats',
        act_quant_percentile=99.999,
        act_quant_type='sym',
        quant_format='int')

    if 'mha' in model_name:
        inp = torch.randn(256, *IN_SIZE_LINEAR[1:])
    else:
        inp = torch.randn(256, *IN_SIZE_CONV[1:])

    dataset = TensorDataset(inp, inp)
    calibloader = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True, shuffle=True)

    apply_gpfq(
        calibloader,
        model,
        act_order=act_order,
        accumulator_bit_width=acc_bit_width,
        a2q_layer_filter_fnc=custom_layer_filter_fnc)


@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("acc_bit_width", [32, 24, 16])
def test_toymodels(act_order: bool, acc_bit_width: int):

    model = QuantConvModel()
    inp = torch.randn(256, *IN_SIZE_CONV[1:])
    dataset = TensorDataset(inp, inp)
    calibloader = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True, shuffle=True)
    apply_gpfq(
        calibloader,
        model,
        act_order=act_order,
        accumulator_bit_width=acc_bit_width,
        a2q_layer_filter_fnc=custom_layer_filter_fnc)
