# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import gptq_mode
from brevitas_examples.imagenet_classification.ptq.ptq_common import _a2q_layer_filter_fnc


from .equalization_fixtures import *

def apply_gpfq(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        max_accumulator_bit_width: int,
        max_accumulator_tile_size: int):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gpfq_mode(model,
                       act_order=act_order,
                       a2q_layer_filter_fnc=_a2q_layer_filter_fnc,
                       use_quant_activations=use_quant_activations,
                       max_accumulator_tile_size=max_accumulator_tile_size,
                       max_accumulator_bit_width=max_accumulator_bit_width) as gpfq:
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
        max_accumulator_bit_width: int,
        max_accumulator_tile_size: int):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gptq_mode(model,
                       act_order=act_order,
                       a2q_layer_filter_fnc=_a2q_layer_filter_fnc,
                       use_quant_activations=use_quant_activations,
                       max_accumulator_bit_width=max_accumulator_bit_width,
                       max_accumulator_tile_size=max_accumulator_tile_size) as gptq:
            gptq_model = gptq.model
            for _ in range(gptq.num_layers):
                for _, (images, _) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gptq_model(images)
                gptq.update()


apply_gpxq_func_map = {"gpfq": apply_gpfq, "gptq": apply_gptq}


@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("use_quant_activations", [True, False])
@pytest.mark.parametrize(
    "apply_gpxq_tuple", apply_gpxq_func_map.items(), ids=apply_gpxq_func_map.keys())
@pytest.mark.parametrize("max_accumulator_bit_width", [None, 12, 32])
@pytest.mark.parametrize("max_accumulator_tile_size", [None, 32])
def test_toymodels(toy_quant_model, act_order, use_quant_activations, apply_gpxq_tuple, max_accumulator_bit_width, max_accumulator_tile_size, request):

    test_id = request.node.callspec.id
    input_quant = test_id.split('-')[1]

    torch.manual_seed(SEED)

    if (max_accumulator_bit_width is None) and (max_accumulator_tile_size is not None):
        pytest.skip("max_accumulator_tile_size doesn't matter if max_accumulator_bit_width is None.")

    if (max_accumulator_bit_width is not None) and input_quant.startswith("MXFloat"):
        pytest.skip("AXE does not currently support minifloat formats.")

    name, apply_gpxq = apply_gpxq_tuple

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
    
    if (max_accumulator_bit_width is not None) and (input_quant == 'None' or not use_quant_activations):
        # AXE (or A2GPxQ) requires that the quant activations are used. A2GPxQ.single_layer_update
        # will raise a ValueError if AXE.quant_metadata is None (also see GPxQ.process_input). This
        # will happen when `use_quant_activations=False` or when the input to a model is not quantized
        # and `a2q_layer_filter_fnc` does not properly handle it.
        with pytest.raises(ValueError):
            apply_gpxq(
                calib_loader=calib_loader,
                model=model,
                act_order=act_order,
                use_quant_activations=use_quant_activations,
                max_accumulator_bit_width=max_accumulator_bit_width,
                max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        apply_gpxq(
            calib_loader=calib_loader,
            model=model,
            act_order=act_order,
            use_quant_activations=use_quant_activations,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
