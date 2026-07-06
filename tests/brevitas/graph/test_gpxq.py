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
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.magr import magr_mode
from brevitas.graph.qronos import Qronos
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat

from .equalization_fixtures import *


@torch.no_grad()
def _dual_optimization_callback(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        algorithm_impl: nn.Module):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gpfq_mode(model,
                       use_quant_activations=use_quant_activations,
                       act_order=act_order,
                       algorithm_impl=algorithm_impl) as algo:
            algo_model = algo.model
            for _ in range(algo.num_layers):
                for _, (images, _) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    algo_model(images)
                algo.update()


def apply_gpfq(
        calib_loader: DataLoader, model: nn.Module, act_order: bool, use_quant_activations: bool):
    _dual_optimization_callback(
        calib_loader=calib_loader,
        model=model,
        act_order=act_order,
        use_quant_activations=use_quant_activations,
        algorithm_impl=GPFQ)


def apply_qronos(
        calib_loader: DataLoader, model: nn.Module, act_order: bool, use_quant_activations: bool):
    _dual_optimization_callback(
        calib_loader=calib_loader,
        model=model,
        act_order=act_order,
        use_quant_activations=use_quant_activations,
        algorithm_impl=Qronos)


def apply_gptq(
        calib_loader: DataLoader, model: nn.Module, act_order: bool, use_quant_activations: bool):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.no_grad():
        with gptq_mode(model, use_quant_activations=use_quant_activations,
                       act_order=act_order) as gptq:
            gptq_model = gptq.model
            for _ in range(gptq.num_layers):
                for _, (images, _) in enumerate(calib_loader):
                    images = images.to(device)
                    images = images.to(dtype)
                    gptq_model(images)
                gptq.update()


apply_gpxq_func_map = {"gpfq": apply_gpfq, "gptq": apply_gptq, "qronos": apply_qronos}


class TestQronosUpdateBatch:
    """Tests for Qronos.update_batch verifying correct H and G normalization."""

    INP = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

    @staticmethod
    def _make_model():
        """Two QuantLinear layers (2→3→4) with hardcoded weights."""

        class Model(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear_0 = qnn.QuantLinear(
                    2, 3, bias=False, weight_quant=Int8WeightPerTensorFloat)
                self.linear_1 = qnn.QuantLinear(
                    3, 4, bias=False, weight_quant=Int8WeightPerTensorFloat)

            def forward(self, x):
                return self.linear_1(self.linear_0(x))

        model = Model()
        with torch.no_grad():
            model.linear_0.weight.copy_(torch.tensor([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4]]))
            model.linear_1.weight.copy_(
                torch.tensor([[0.5, -0.3, 0.1], [0.2, 0.4, -0.2], [-0.1, 0.3, 0.5],
                              [0.4, -0.1, 0.2]]))
        return model

    @staticmethod
    def _calibrate(model, calib_loader):
        """Run Qronos calibration, return {layer_name: (H, G)} for each layer."""
        results = {}
        with torch.no_grad():
            with gpfq_mode(model, act_order=False, algorithm_impl=Qronos) as algo:
                for _ in range(algo.num_layers):
                    for data, _ in calib_loader:
                        algo.model(data)
                    for name in algo.current_layer.layer_names:
                        layer = algo.gpxq_layers[name]
                        results[name] = (layer.H.clone(), layer.G.clone())
                    algo.update()
        return results

    def _make_loader(self, batch_size):
        dataset = TensorDataset(self.INP, self.INP)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def _init_model(self):
        model = self._make_model()
        model.eval()
        model(self.INP)  # collect scaling factors
        return model

    def test_h_and_g_values(self):
        """Verify H and G have the correct analytical values.

        For linear_0 (first layer), the input is the known external input in both the
        quant and float passes (since input_quant=None), so H == G == X @ X.T / N.

        For linear_1, intermediate activations differ between the quant and float passes
        (weights are quantized in one, float in the other), so H != G in general.
        We verify H is symmetric (X̂ @ X̂.T) and that G is non-zero.

        Current convention on dev: G = X @ X̂.T / N (float @ quant.T).
        TODO: If https://github.com/Xilinx/brevitas/pull/1501 is merged, G convention
        changes to X̂ @ X.T / N (quant @ float.T) and this test must be updated.
        """
        results = self._calibrate(self._init_model(), self._make_loader(batch_size=4))

        # linear_0: no input quant, so quant_input == float_input == X
        x = self.INP.t().unsqueeze(0).float()  # [1, in_features, N]
        expected = x.bmm(x.transpose(2, 1)) / 4  # X @ X.T / N
        H0, G0 = results['linear_0']
        torch.testing.assert_close(H0, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(G0, expected, atol=1e-5, rtol=1e-5)

        # linear_1: H should be symmetric, both H and G should be non-zero
        H1, G1 = results['linear_1']
        torch.testing.assert_close(H1, H1.transpose(1, 2), atol=1e-6, rtol=1e-6)
        assert H1.abs().sum() > 0
        assert G1.abs().sum() > 0

    def test_multi_batch_normalization(self):
        """Runs calibration with 1 batch of 4 vs 2 batches of 2 and asserts H and G
        are identical for both layers, verifies that the running-average normalization
        is batch-size invariant."""
        results_single = self._calibrate(self._init_model(), self._make_loader(batch_size=4))
        results_multi = self._calibrate(self._init_model(), self._make_loader(batch_size=2))

        for name in results_single:
            H_s, G_s = results_single[name]
            H_m, G_m = results_multi[name]
            torch.testing.assert_close(H_s, H_m, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(G_s, G_m, atol=1e-5, rtol=1e-5)

    def test_no_inplace_input_mutation(self):
        """Clones the input before each forward pass and asserts it was not modified,
        catching any in-place normalization (e.g. /=) in update_batch that would
        corrupt inputs."""
        model = self._init_model()
        with torch.no_grad():
            with gpfq_mode(model, act_order=False, algorithm_impl=Qronos) as algo:
                for _ in range(algo.num_layers):
                    for data, _ in self._make_loader(batch_size=2):
                        data_before = data.clone()
                        algo.model(data)
                        torch.testing.assert_close(data, data_before)


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

    apply_gpxq(
        calib_loader=calib_loader,
        model=model,
        act_order=act_order,
        use_quant_activations=use_quant_activations)


@torch.no_grad()
def apply_magr(
        model,
        dataloader,
        create_weight_orig=False,
        group_of_parallel_layers=None,
        alpha=0.1,
        num_steps=10):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with magr_mode(model,
                   group_of_parallel_layers=group_of_parallel_layers,
                   create_weight_orig=create_weight_orig,
                   num_steps=num_steps,
                   alpha=alpha) as magr:
        magr_model = magr.model
        for _, (images, _) in enumerate(dataloader):
            images = images.to(device)
            images = images.to(dtype)
            magr_model(images)
        magr.update()


def test_magr(toy_model, request):
    test_id = request.node.callspec.id

    torch.manual_seed(SEED)

    model_class = toy_model
    model = model_class()
    if 'mha' in test_id:
        inp = torch.randn(32, *IN_SIZE_LINEAR[1:])
    else:
        inp = torch.randn(32, *IN_SIZE_CONV_SMALL[1:])
    model.eval()
    model(inp)  # test forward pass and collect scaling factors
    dataset = TensorDataset(inp, inp)
    dataloader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=True, shuffle=True)

    apply_magr(model, dataloader)
