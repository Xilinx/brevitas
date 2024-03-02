# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import gptq_mode
import brevitas.nn as qnn


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
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool = True,
        accumulator_bit_width: int = 32,
        a2q_layer_filter_fnc=lambda x: True):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        # use A2GPFQ if accumulator is less than 32 is specified
        with gpfq_mode(
                model,
                use_quant_activations=use_quant_activations,
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


def apply_gptq(
        calib_loader: DataLoader, model: nn.Module, act_order: bool, use_quant_activations: bool):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gptq_mode(
                model,
                use_quant_activations=use_quant_activations,
                act_order=act_order,
        ) as gptq:
            gptq_model = gptq.model
            for _ in range(gptq.num_layers):
                for _, (images, _) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gptq_model(images)
                gptq.update()


def custom_layer_filter_fnc(layer: nn.Module) -> bool:
    if isinstance(layer, nn.Conv2d) and layer.in_channels == 3:
        return False
    return True


def identity_layer_filter_func(layer: nn.Module) -> bool:
    return True


filter_func_dict = {
    "identity": identity_layer_filter_func,
    "ignore_input": custom_layer_filter_fnc,}


@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("use_quant_activations", [True, False])
@pytest.mark.parametrize("acc_bit_width", [32, 24, 16, 12])
@pytest.mark.parametrize("filter_func_str", filter_func_dict.keys())
def test_gpfq(
        act_order: bool, use_quant_activations: bool, acc_bit_width: int, filter_func_str: str):
    model = QuantConvModel()
    inp = torch.randn(100, 3, 32, 32)
    dataset = TensorDataset(inp, inp)
    calibloader = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True, shuffle=True)
    filter_func = filter_func_dict[filter_func_str]
    if (acc_bit_width < 32) and (not use_quant_activations or filter_func_str == "identity"):
        # GPFA2Q requires that the quant activations are used. GPFA2Q.single_layer_update will
        # raise a ValueError if GPFA2Q.quant_input is None (also see GPxQ.process_input). This will
        # happen when `use_quant_activations=False` or when the input to a model is not quantized
        # and `a2q_layer_filter_fnc` does not properly handle it.
        with pytest.raises(ValueError):
            apply_gpfq(
                calibloader,
                model,
                act_order=act_order,
                use_quant_activations=use_quant_activations,
                accumulator_bit_width=acc_bit_width,
                a2q_layer_filter_fnc=filter_func)
    else:
        apply_gpfq(
            calibloader,
            model,
            act_order=act_order,
            use_quant_activations=use_quant_activations,
            accumulator_bit_width=acc_bit_width,
            a2q_layer_filter_fnc=filter_func)


@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("use_quant_activations", [True, False])
def test_gptq(act_order: bool, use_quant_activations: bool):
    model = QuantConvModel()
    inp = torch.randn(100, 3, 32, 32)
    dataset = TensorDataset(inp, inp)
    calibloader = DataLoader(dataset, batch_size=32, num_workers=0, pin_memory=True, shuffle=True)
    apply_gptq(calibloader, model, act_order=act_order, use_quant_activations=use_quant_activations)
