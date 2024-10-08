# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gpfq import GPFQv2
from brevitas.graph.gptq import gptq_mode

from .equalization_fixtures import *


def apply_gpfq(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        gpfq_class: GPFQ):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gpfq_mode(model,
                       use_quant_activations=use_quant_activations,
                       act_order=act_order,
                       gpfq_class=gpfq_class) as gpfq:
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


apply_gpxq_func_map = {
    "gpfq": partial(apply_gpfq, gpfq_class=GPFQ),
    "gpfq2": partial(apply_gpfq, gpfq_class=GPFQv2),
    "gptq": apply_gptq}


@pytest.mark.parametrize("act_order", [True, False])
@pytest.mark.parametrize("use_quant_activations", [True, False])
@pytest.mark.parametrize(
    "apply_gpxq_tuple", apply_gpxq_func_map.items(), ids=apply_gpxq_func_map.keys())
def test_toymodels(toy_quant_model, act_order, use_quant_activations, apply_gpxq_tuple, request):

    test_id = request.node.callspec.id

    torch.manual_seed(SEED)

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

    if ((name == 'gptq' or name == 'gpfq2') and torch_version < version.parse('1.10')):
        # Usage of linalg_cholesky() is not compatible with torch 1.9.1 and below
        with pytest.raises(AssertionError):
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
