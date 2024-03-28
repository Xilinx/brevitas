# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from typing import Optional

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import gptq_mode
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from brevitas.utils.quant_utils import calculate_accumulator_bit_width

from .equalization_fixtures import *


def apply_gpfq(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool = True,
        accumulator_bit_width: Optional[int] = None,
        a2q_layer_filter_fnc=lambda x: False):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gpfq_mode(
                model,
                use_quant_activations=use_quant_activations,
                act_order=act_order,
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
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        accumulator_bit_width: Optional[int] = None,
        a2q_layer_filter_fnc=lambda x: False):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gptq_mode(
                model,
                use_quant_activations=use_quant_activations,
                act_order=act_order,
                accumulator_bit_width=accumulator_bit_width,
                a2q_layer_filter_fnc=a2q_layer_filter_fnc,
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
    elif isinstance(layer, nn.ConvTranspose2d) and layer.in_channels == 3:
        return False
    return True


def identity_layer_filter_func(layer: nn.Module, enable: bool) -> bool:
    return enable


filter_func_dict = {
    "identity_enabled": partial(identity_layer_filter_func, enable=True),
    "identity_disabled": partial(identity_layer_filter_func, enable=False),
    "ignore_input": custom_layer_filter_fnc}

apply_gpxq_func_map = {"gpfq": apply_gpfq, "gptq": apply_gptq}


@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("use_quant_activations", [True, False])
@pytest.mark.parametrize("acc_bit_width", [None, 24, 16, 12])
@pytest.mark.parametrize("filter_func_str", filter_func_dict.keys())
@pytest.mark.parametrize("apply_gpxq_tuple", apply_gpxq_func_map.items())
def test_toymodels(
        toy_quant_model,
        act_order,
        use_quant_activations,
        acc_bit_width,
        filter_func_str,
        apply_gpxq_tuple,
        request):

    test_id = request.node.callspec.id

    torch.manual_seed(SEED)

    if (filter_func_str == 'identity_disabled') and (acc_bit_width
                                                     is not None) and (acc_bit_width < 32):
        pytest.skip("These cases disable accumulator-aware quantization.")

    name, apply_gpxq = apply_gpxq_tuple

    filter_func = filter_func_dict[filter_func_str]
    apply_gpxq = partial(
        apply_gpxq, accumulator_bit_width=acc_bit_width, a2q_layer_filter_fnc=filter_func)

    model_class = toy_quant_model
    model = model_class()
    if 'mha' in test_id:
        inp = torch.randn(32, *IN_SIZE_LINEAR[1:])
    else:
        inp = torch.randn(32, *IN_SIZE_CONV_SMALL[1:])
    model.eval()
    model(inp)  # test forward pass and collect scaling factors
    dataset = TensorDataset(inp, inp)
    calib_loader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=True, shuffle=True)

    if (name == 'gptq' and torch_version < version.parse('1.10')):
        # GPTQ usage of linalg_cholesky() is not compatible with torch 1.9.1 and below
        with pytest.raises(AssertionError):
            apply_gpxq(
                calib_loader=calib_loader,
                model=model,
                act_order=act_order,
                use_quant_activations=use_quant_activations)

    elif (acc_bit_width is not None) and (not use_quant_activations or
                                          filter_func_str == 'identity_enabled'):
        # `A2GPFxQ` requires that the quant activations are used. `A2GPFxQ.single_layer_update(...)` will
        # raise a ValueError if `A2GPFxQ.quant_input` is None (also see `A2GPxQ.process_input`). This will
        # happen when `use_quant_activations=False` or when the input to a model is not quantized
        # and `a2q_layer_filter_fnc` does not properly handle it.
        with pytest.raises(ValueError):
            apply_gpxq(
                calib_loader=calib_loader,
                model=model,
                act_order=act_order,
                use_quant_activations=use_quant_activations)
    else:
        apply_gpxq(
            calib_loader=calib_loader,
            model=model,
            act_order=act_order,
            use_quant_activations=use_quant_activations)

        for module in model.modules():
            if isinstance(module, QuantWBIOL):
                module.cache_inference_quant_inp = True

        model(inp)  # collect inputs

        if acc_bit_width is not None:
            for name, module in model.named_modules():
                if filter_func(module) and isinstance(module, QuantWBIOL):
                    cur_acc_bit_width = calculate_accumulator_bit_width(module)
                    assert cur_acc_bit_width <= acc_bit_width, f"Error in {name}: {cur_acc_bit_width} <= {acc_bit_width}"
