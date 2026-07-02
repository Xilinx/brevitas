# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.gpxq import gpxq_mode
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.magr import magr_mode
from brevitas.graph.qronos import Qronos
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas_examples.common.axe import a2gpfq_mode
from brevitas_examples.common.axe import a2gptq_mode
from brevitas_examples.common.axe import AXEMixin

from .equalization_fixtures import *


def _a2q_layer_filter_fnc(layer: nn.Module) -> bool:
    if isinstance(layer, nn.Conv2d):
        # Skip when columns == 1 (kernel_size=1 and depthwise)
        kernel_size = np.prod(layer.kernel_size)
        if kernel_size == 1 and layer.groups == layer.in_channels:
            return False
    # Known issue with ConvTranspose2d (#1479)
    if isinstance(layer, nn.ConvTranspose2d):
        return False
    return gpxq_mode._is_module_supported(None, layer)


def _verify_accumulator_constraints(gpxq_impl, max_accumulator_bit_width):
    # Independently recompute the worst-case signed integer accumulator from the final quantized
    # weights and assert it fits the budget. This is the inference-time guarantee AXE exists to
    # provide, checked without relying on AXE's internal accumulator tracking. We reuse the AXE
    # instance's own input bounds and weight unrolling so we check against exactly what it
    # constrained against.
    max_limit = 2 ** (max_accumulator_bit_width - 1) - 1
    input_max, input_min = gpxq_impl.input_max, gpxq_impl.input_min
    tile_size = gpxq_impl.max_accumulator_tile_size
    # weights as integers, unrolled to [OC, columns] the same way AXE does
    weight = gpxq_impl.reshape_gpxq_weights(gpxq_impl.layer.quant_weight().int().float())
    columns = weight.shape[1]
    for i in range(0, columns, tile_size):
        tile = weight[:, i:i + tile_size]
        pos = torch.clamp_min(tile, 0).sum(dim=1)  # [OC]
        neg = torch.clamp_max(tile, 0).sum(dim=1)  # [OC]
        pos_acc = (input_max * pos + input_min * neg).max().item()
        neg_acc = (-(input_min * pos + input_max * neg)).max().item()
        assert pos_acc <= max_limit, f"positive accumulator {pos_acc} exceeds {max_limit}"
        assert neg_acc <= max_limit, f"negative accumulator {neg_acc} exceeds {max_limit}"


@torch.no_grad()
def _dual_optimization_callback(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        algorithm_impl: nn.Module,
        max_accumulator_bit_width: int = None,
        max_accumulator_tile_size: int = None):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    context_manager_kwargs = dict(
        model=model,
        use_quant_activations=use_quant_activations,
        act_order=act_order,
        algorithm_impl=algorithm_impl)
    context_manager = gpfq_mode
    if max_accumulator_bit_width is not None:
        context_manager = a2gpfq_mode
        context_manager_kwargs.update(
            a2q_layer_filter_fnc=_a2q_layer_filter_fnc,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    with context_manager(**context_manager_kwargs) as algo:
        algo_model = algo.model
        for _ in range(algo.num_layers):
            for _, (images, _) in enumerate(calib_loader):
                images = images.to(device)
                images = images.to(dtype)
                algo_model(images)
            algo.update()
        if max_accumulator_bit_width is not None:
            # gpxq_layers mixes AXE and plain GPxQ instances (layers failing the a2q filter fall
            # back to the base class); only the AXE instances carry accumulator constraints.
            n_verified = 0
            for gpxq_impl in algo.gpxq_layers.values():
                if isinstance(gpxq_impl, AXEMixin):
                    _verify_accumulator_constraints(gpxq_impl, max_accumulator_bit_width)
                    n_verified += 1
            # guard against silently verifying nothing (e.g. if no layer became an AXE instance)
            assert n_verified > 0, "AXE was enabled but no layer was accumulator-constrained"


def apply_gpfq(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        max_accumulator_bit_width: int = None,
        max_accumulator_tile_size: int = None):
    _dual_optimization_callback(
        calib_loader=calib_loader,
        model=model,
        act_order=act_order,
        use_quant_activations=use_quant_activations,
        algorithm_impl=GPFQ,
        max_accumulator_bit_width=max_accumulator_bit_width,
        max_accumulator_tile_size=max_accumulator_tile_size)


def apply_qronos(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        max_accumulator_bit_width: int = None,
        max_accumulator_tile_size: int = None):
    assert max_accumulator_bit_width is None
    assert max_accumulator_tile_size is None
    _dual_optimization_callback(
        calib_loader=calib_loader,
        model=model,
        act_order=act_order,
        use_quant_activations=use_quant_activations,
        algorithm_impl=Qronos)


@torch.no_grad()
def apply_gptq(
        calib_loader: DataLoader,
        model: nn.Module,
        act_order: bool,
        use_quant_activations: bool,
        max_accumulator_bit_width: int = None,
        max_accumulator_tile_size: int = None):
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    context_manager_kwargs = dict(
        model=model, act_order=act_order, use_quant_activations=use_quant_activations)
    context_manager = gptq_mode
    if max_accumulator_bit_width is not None:
        context_manager = a2gptq_mode
        context_manager_kwargs.update(
            a2q_layer_filter_fnc=_a2q_layer_filter_fnc,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    with context_manager(**context_manager_kwargs) as gptq:
        gptq_model = gptq.model
        for _ in range(gptq.num_layers):
            for _, (images, _) in enumerate(calib_loader):
                images = images.to(device)
                images = images.to(dtype)
                gptq_model(images)
            gptq.update()
        if max_accumulator_bit_width is not None:
            # gpxq_layers mixes AXE and plain GPxQ instances (layers failing the a2q filter fall
            # back to the base class); only the AXE instances carry accumulator constraints.
            n_verified = 0
            for gpxq_impl in gptq.gpxq_layers.values():
                if isinstance(gpxq_impl, AXEMixin):
                    _verify_accumulator_constraints(gpxq_impl, max_accumulator_bit_width)
                    n_verified += 1
            # guard against silently verifying nothing (e.g. if no layer became an AXE instance)
            assert n_verified > 0, "AXE was enabled but no layer was accumulator-constrained"


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
@pytest.mark.parametrize("max_accumulator_bit_width", [None, 12, 32])
@pytest.mark.parametrize("max_accumulator_tile_size", [None, 32])
def test_toy_quant_models(
        toy_quant_model,
        act_order,
        use_quant_activations,
        apply_gpxq_tuple,
        max_accumulator_bit_width,
        max_accumulator_tile_size,
        request):

    test_id = request.node.callspec.id
    input_quant = test_id.split('-')[1]

    torch.manual_seed(SEED)

    if (max_accumulator_bit_width is None) and (max_accumulator_tile_size is not None):
        pytest.skip(
            "max_accumulator_tile_size doesn't matter if max_accumulator_bit_width is None.")

    if (max_accumulator_bit_width is not None) and input_quant.startswith("MXFloat"):
        pytest.skip("No support for AXE + Float.")

    name, apply_gpxq = apply_gpxq_tuple

    if (max_accumulator_bit_width is not None) and (name == "qronos"):
        pytest.skip("No support for AXE + Qronos.")

    model_class = toy_quant_model
    model = model_class()

    gpxq_layers = [mod for mod in model.modules() if _a2q_layer_filter_fnc(mod)]
    if max_accumulator_bit_width is not None and not gpxq_layers:
        pytest.skip(f"AXE does not support any modules in {name}.")

    inp = torch.randn(32, *model.input_size)
    model.eval()
    model(inp)  # test forward pass and collect scaling factors
    dataset = TensorDataset(inp, inp)
    calib_loader = DataLoader(dataset, batch_size=16, num_workers=0, pin_memory=True, shuffle=True)

    def _is_value_error_expected():
        # The conditions below only matter for AXE (A2GPxQ); plain GPxQ has no such constraints
        if max_accumulator_bit_width is None:
            return False
        # AXE needs quantized activation metadata to compute the accumulator bounds. With no input
        # quantizer, AXE.quant_metadata is None and A2GPxQ.single_layer_update raises the exception
        if input_quant == 'None':
            return True
        # Same failure for a different reason: leaving activations unquantized during GPxQ means the
        # quantized input metadata is never captured, so AXE.quant_metadata is None
        if not use_quant_activations:
            return True
        # AXE only supports groupwise weight scales for linear layers; the AXEMixin constructor
        # rejects groupwise weight quantization on convolutions
        for mod in gpxq_layers:
            if mod.weight_quant.is_groupwise and isinstance(mod, SUPPORTED_CONV_OP):
                return True
        return False

    if _is_value_error_expected():
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


class _MockAXEMixin(AXEMixin):
    # Minimal AXEMixin host that exposes get_thresholds without a real layer or context manager.
    # We bypass AXEMixin.__init__ (which needs a layer) and set only what get_thresholds reads.
    def __init__(self, max_accumulator_bit_width, max_accumulator_tile_size, input_bit_width):
        self.max_accumulator_bit_width = torch.tensor(float(max_accumulator_bit_width))
        self.max_accumulator_tile_size = max_accumulator_tile_size
        self.groups = 1
        self._input_max = 2 ** (input_bit_width - 1) - 1
        self._input_min = -2 ** (input_bit_width - 1)

    @property
    def input_max(self):
        return self._input_max

    @property
    def input_min(self):
        return self._input_min

    @property
    def radius(self):
        # L1-ball radius (the per-tile accumulator budget in the integer domain)
        return (2 ** self.max_accumulator_bit_width - 2) / float(self.input_max - self.input_min)


class TestAXEThresholds:
    # get_thresholds must project the zero-centered integer-domain weights (w / s) onto an L1 ball
    # of the accumulator budget radius, per tile, then rescale into the float domain. Each test
    # builds weights/scales with a known closed-form oracle and compares against get_thresholds.
    #
    # Monolithic 16-bit accumulator, 8-bit signed input -> radius = (2**16 - 2) / 255 ~= 257.
    accumulator_bit_width = 16
    input_bit_width = 8
    eps = 1e-8

    @property
    def radius(self):
        return (2 ** self.accumulator_bit_width - 2) / (2 ** self.input_bit_width - 1)

    @staticmethod
    def _l1_ball_threshold(a, n, radius):
        # Closed-form soft-threshold for a vector of `n` EQUAL nonzero magnitudes `a` projected
        # onto an L1 ball: 0 if already inside (n * a <= radius), else a - radius / n.
        if n * a <= radius:
            return 0.0
        return a - radius / n

    @staticmethod
    def _expand_group_scales(tile_scales, tile_size, in_features):
        # Expand one scale per tile [OC, n_tiles] to per-input-channel [OC, in_features], mirroring
        # how Brevitas expands a compact groupwise scale back to the weight shape: repeat each
        # group's scale across the group, then slice off the padding down to the real in_features
        # (see brevitas.utils.quant_utils.groupwise_dequant_expand).
        out_features, n_tiles = tile_scales.shape
        scales = tile_scales.unsqueeze(-1).expand(out_features, n_tiles, tile_size)
        return scales.reshape(out_features, n_tiles * tile_size)[:, :in_features]

    def _build_equal_magnitude_case(self, out_features, in_features, tile_size, alpha):
        # Every element has integer-domain magnitude |alpha| (constant), alternating sign for zero
        # mean per tile. Each tile gets its own random positive scale (groupwise along the input
        # dim); the float weight is (integer * scale) so get_thresholds recovers |alpha| after w / s.
        # The last tile may be short (ragged), which get_thresholds pads internally. Returns
        # weight/scales [OC, IC] and the closed-form oracle thresholds [1, n_tiles, OC].
        n_tiles = math.ceil(in_features / tile_size)
        last_tile_size = tile_size if in_features % tile_size == 0 else in_features % tile_size
        assert tile_size % 2 == 0 and last_tile_size % 2 == 0, \
            "each tile needs an even width for the alternating-sign zero mean to hold"

        tile_scales = torch.rand(out_features, n_tiles) + self.eps
        scales = self._expand_group_scales(tile_scales, tile_size, in_features)

        int_weight = torch.full((out_features, in_features), float(alpha))
        int_weight[:, 1::2] *= -1  # alternating sign -> zero mean within every (even-width) tile
        weight = int_weight * scales

        expected = tile_scales.clone() * self._l1_ball_threshold(alpha, tile_size, self.radius)
        # the (possibly ragged) last tile has fewer real elements, so its threshold differs
        expected[:, -1] = tile_scales[:, -1] * self._l1_ball_threshold(
            alpha, last_tile_size, self.radius)
        expected = expected.transpose(0, 1).unsqueeze(0)  # [1, n_tiles, OC]
        return weight, scales, expected

    def _run(self, weight, scales, tile_size, expected):
        axe = _MockAXEMixin(self.accumulator_bit_width, tile_size, self.input_bit_width)
        n_tiles = math.ceil(weight.shape[-1] / tile_size)
        # get_thresholds expects [groups, OC/groups, IC]; groups=1 so add a leading singleton dim
        thresholds = axe.get_thresholds(weight.unsqueeze(0), scales.unsqueeze(0), n_tiles)
        assert thresholds.shape == expected.shape
        assert torch.allclose(thresholds, expected, atol=1e-5, rtol=1e-4)

    # in_features covers a single tile (16), a ragged/padded last tile (24 -> 16 + 8), and multiple
    # full tiles (32 -> 16 + 16). Per-tile random scales exercise the groupwise scale mapping.
    @pytest.mark.parametrize("in_features", [16, 24, 32])
    def test_outside_ball(self, in_features, out_features=2, tile_size=16, offset=10):
        # every tile outside the ball (n * alpha > radius for all tile widths n) -> theta > 0.
        # size alpha off the smallest tile so the short/ragged tile is outside too.
        n = tile_size if in_features % tile_size == 0 else in_features % tile_size
        alpha = self.radius / n + offset
        weight, scales, expected = self._build_equal_magnitude_case(
            out_features, in_features, tile_size, alpha)
        assert (expected > 0).all()  # confirm we actually exercised the projection
        self._run(weight, scales, tile_size, expected)

    @pytest.mark.parametrize("in_features", [16, 24, 32])
    def test_inside_ball(self, in_features, out_features=2, tile_size=16):
        # every tile inside the ball (n * alpha <= radius for all tile widths n) -> theta == 0
        alpha = self.radius / (2 * tile_size)
        weight, scales, expected = self._build_equal_magnitude_case(
            out_features, in_features, tile_size, alpha)
        assert (expected == 0).all()  # confirm the no-shrinkage branch
        self._run(weight, scales, tile_size, expected)

    def test_unequal_magnitudes(self):
        # hand-solved oracle exercising the sort/threshold-search path that equal magnitudes cannot.
        # accumulator_bit_width=5, input_bit_width=2 -> radius = (2**5 - 2) / (2**2 - 1) = 30/3 = 10.
        # tile (w / s) = [8, 4, -1, -11], mean 0, |v| = [8, 4, 1, 11]. Projecting onto radius 10
        # keeps the top two {11, 8}: theta = (11 + 8 - 10) / 2 = 4.5.
        accumulator_bit_width, input_bit_width, tile_size = 5, 2, 4
        s = 0.25
        weight = (torch.tensor([8.0, 4.0, -1.0, -11.0]) * s).view(1, 4)  # [OC=1, IC=4]
        scales = torch.full((1, 4), s)
        expected = torch.tensor(4.5 * s).view(1, 1, 1)  # [1, n_tiles=1, OC=1]
        axe = _MockAXEMixin(accumulator_bit_width, tile_size, input_bit_width)
        thresholds = axe.get_thresholds(weight.unsqueeze(0), scales.unsqueeze(0), 1)
        assert thresholds.shape == expected.shape
        assert torch.allclose(thresholds, expected, atol=1e-5, rtol=1e-4)

    def test_nonzero_mean(self):
        # hand-solved oracle exercising the zero-centering step (every other case has mean 0).
        # accumulator_bit_width=5, input_bit_width=2 -> radius = (2**5 - 2) / (2**2 - 1) = 30/3 = 10.
        # tile (w / s) = [10, 6, 2, 2], mean 5 -> centered [5, 1, -3, -3], |v| = [5, 1, 3, 3], sum 12
        # > 10. All four survive: theta = (12 - 10) / 4 = 0.5.
        accumulator_bit_width, input_bit_width, tile_size = 5, 2, 4
        s = 0.25
        weight = (torch.tensor([10.0, 6.0, 2.0, 2.0]) * s).view(1, 4)  # [OC=1, IC=4]
        scales = torch.full((1, 4), s)
        expected = torch.tensor(0.5 * s).view(1, 1, 1)  # [1, n_tiles=1, OC=1]
        axe = _MockAXEMixin(accumulator_bit_width, tile_size, input_bit_width)
        thresholds = axe.get_thresholds(weight.unsqueeze(0), scales.unsqueeze(0), 1)
        assert thresholds.shape == expected.shape
        assert torch.allclose(thresholds, expected, atol=1e-5, rtol=1e-4)
