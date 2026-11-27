# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
from contextlib import contextmanager
import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.piso import ScaleOptimizer
from brevitas.graph.qronos import Qronos
from brevitas.graph.utils import get_module_name_and_parent
from brevitas.graph.utils import gpxq_compute_error_stats
from brevitas.graph.utils import set_module
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.utils.stats_utils import collect_stats
from brevitas.utils.stats_utils import DictStatsCollector
from brevitas_examples.llm.llm_quant.gpxq import apply_gptq
from brevitas_examples.llm.llm_quant.gpxq import apply_gpxq_stats
from brevitas_examples.llm.llm_quant.gpxq import apply_qronos
from brevitas_examples.llm.llm_quant.piso import apply_scale_optimization
from tests.conftest import SEED

IN_FEATURES = 2
OUT_FEATURES = 2

torch.manual_seed(SEED)


class SubModel(nn.Module):

    def __init__(self):
        super(SubModel, self).__init__()
        self.linear = nn.Linear(IN_FEATURES, OUT_FEATURES)

    def forward(self, x):
        return self.linear(x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.sub_model = SubModel()

    def forward(self, x):
        return self.sub_model(x)


def test_get_module_name_and_parent():
    model = Model()
    module_name, supermodule = get_module_name_and_parent(model, "sub_model.linear")
    assert module_name == "linear"
    assert supermodule is model.sub_model


def test_set_module():
    model = Model()
    new_module = nn.Linear(IN_FEATURES, OUT_FEATURES)
    set_module(model, new_module, "sub_model.linear")
    assert model.sub_model.linear is new_module


# ---------------------------------------------------------------------------
# Tests for the PTQ error statistics
# ---------------------------------------------------------------------------

STATS_IN_FEATURES = 4
STATS_HIDDEN_FEATURES = 8
STATS_SEQLEN = 16
STATS_NUM_SAMPLES = 8
STATS_SEED = 42

# The layer the per-layer assertions target. It is the *second* layer of the first
# block, so its quantized and float inputs genuinely differ (they are produced by
# the preceding layer with quantized vs float weights) in both the block-wise and
# the whole-model drivers.
STATS_LAYER = "blocks.0.linear_2"

# Per-algorithm metadata for the unified statistics tests.
#   cov_scale   : normalization used when accumulating the covariances (1/T vs 2/T).
#   weight_attr : floating-point reference. GPxQ overwrites `weight` and keeps
#                 `weight_orig`; PiSO leaves `weight` untouched.
#   g_transposed: the algorithm accumulates X_tilde^T X instead of X^T X_tilde.
_STATS_ALGO_CONFIG = {
    'gptq': {
        'cov_scale': 2.0, 'weight_attr': 'weight_orig', 'g_transposed': False},
    'qronos': {
        'cov_scale': 1.0, 'weight_attr': 'weight_orig', 'g_transposed': False},
    'sopt': {
        'cov_scale': 2.0, 'weight_attr': 'weight', 'g_transposed': True},}


class QuantLinearBlock(nn.Module):
    """Minimal stand-in for a transformer block. Two layers, so that the second one
    sees different inputs on the quantized and the float calibration pass."""

    def __init__(self, weight_quant):
        super().__init__()
        self.linear_1 = qnn.QuantLinear(
            STATS_IN_FEATURES, STATS_HIDDEN_FEATURES, bias=False, weight_quant=weight_quant)
        self.linear_2 = qnn.QuantLinear(
            STATS_HIDDEN_FEATURES, STATS_IN_FEATURES, bias=False, weight_quant=weight_quant)

    def forward(self, x):
        return self.linear_2(self.linear_1(x))


class QuantLinearModel(nn.Module):
    """Two-block model exposing the attributes the block-wise driver expects."""

    def __init__(self, weight_quant=Int8WeightPerChannelFloat):
        super().__init__()
        weight_quant = weight_quant.let(bit_width=3, scaling_impl_type="parameter_from_stats")
        self.config = Namespace(use_cache=False)
        self.blocks = nn.ModuleList([QuantLinearBlock(weight_quant) for _ in range(2)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


def _build_model_and_calibration_data():
    """Model with initialized scales plus deterministic calibration data of shape
    ``[STATS_NUM_SAMPLES, STATS_SEQLEN, STATS_IN_FEATURES]``."""
    torch.manual_seed(STATS_SEED)
    model = QuantLinearModel()
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, STATS_SEQLEN, STATS_IN_FEATURES))
    torch.manual_seed(STATS_SEED + 1)
    return model, torch.randn(STATS_NUM_SAMPLES, STATS_SEQLEN, STATS_IN_FEATURES)


@contextmanager
def _float_model(model):
    with quantization_status_manager(model,
                                     disable_act_quant=True,
                                     disable_weight_quant=True,
                                     disable_bias_quant=True,
                                     is_training=False):
        yield


def _quant_and_float_outputs(model, module, inp):
    """``(Y_quant, Y_float)``: the output of *module* with quantized and with float
    weights, for the same input."""
    with torch.no_grad():
        y_quant = module(inp)
        with _float_model(model):
            y_float = module(inp)
    return y_quant, y_float


def _expected_covariance(left, right=None, scale=2.0):
    """``scale/T * left^T right``, matching how the algorithms accumulate H/G/R."""
    right = left if right is None else right

    def _prep(x):
        x = x.reshape(-1, x.shape[-1]).to(torch.float32)
        return (math.sqrt(scale / x.shape[0]) * x.t()).unsqueeze(0)  # [1, C, T]

    return _prep(left).bmm(_prep(right).transpose(2, 1))


def _sopt_args(block_name):
    return Namespace(
        input_bit_width=None,
        gpxq_block_name=block_name,
        weight_quant_granularity='per_channel',
        sopt_objective='cross-activation',
        sopt_use_quant_activations=False,
        sopt_solver_batch_size=None,
        sopt_group_sequential=False,
        sopt_group_sequential_act_order=True,
        sopt_hessian_mode='dense',
        sopt_solver='piso')


@contextmanager
def _noop_layer_update(algo):
    """Freeze the per-layer update so that the weights/scales are untouched and the
    pre- and post-update statistics coincide."""
    target = {'gptq': GPTQ, 'qronos': Qronos, 'sopt': ScaleOptimizer}[algo]
    with patch.object(target, 'single_layer_update', lambda self, *a, **kw: None):
        yield


def _run_with_stats(algo, model, calib_input, blockwise, captured=None):
    """Run *algo* with a frozen layer update and return the collected statistics.

    When *captured* is a dict, the raw H/G/R tensors handed to the statistics
    callback are recorded under ``"{layer}_{prefix}_{matrix}"``.
    """
    dataloader = [{"x": calib_input}]
    block_name = "blocks" if blockwise else None

    def _capture(prefix, name, layer, H, G=None, R=None, **kwargs):
        captured[f"{name}_{prefix}_H"] = H.clone()
        if G is not None:
            captured[f"{name}_{prefix}_G"] = G.clone()
        if R is not None:
            captured[f"{name}_{prefix}_R"] = R.clone()
        return gpxq_compute_error_stats(
            prefix=prefix, name=name, layer=layer, H=H, G=G, R=R, **kwargs)

    collector = DictStatsCollector()
    with collect_stats(collector), _noop_layer_update(algo), torch.no_grad():
        # The modes resolve the statistics function when they are constructed, which
        # happens inside the entrypoints below.
        with patch("brevitas.graph.layerwise_hook.gpxq_compute_error_stats",
                   _capture if captured is not None else gpxq_compute_error_stats):
            # Everything runs on CPU in these tests.
            with patch("brevitas_examples.llm.llm_quant.gpxq.send_to_device", lambda x, device: x):
                if algo == 'gptq':
                    apply_gptq(model, dataloader, block_name=block_name, create_weight_orig=True)
                elif algo == 'qronos':
                    apply_qronos(model, dataloader, block_name=block_name)
                else:
                    apply_scale_optimization(model, dataloader, _sopt_args(block_name))
    return collector


class TestPTQStats:

    @staticmethod
    def _target_layer_inputs(model, calib_input):
        """``(X_quant, X_float)`` for STATS_LAYER."""
        return _quant_and_float_outputs(model, model.blocks[0].linear_1, calib_input)

    @pytest.mark.parametrize("algo", ["gptq", "qronos", "sopt"])
    @pytest.mark.parametrize("blockwise", [False, True])
    def test_covariances_captured(self, algo, blockwise):
        """H, G and R handed to the statistics callback must match the covariances
        of the inputs the layer actually saw, with the right orientation for G."""
        cfg = _STATS_ALGO_CONFIG[algo]
        scale = cfg['cov_scale']
        model, calib_input = _build_model_and_calibration_data()
        x_quant, x_float = self._target_layer_inputs(model, calib_input)

        captured = {}
        _run_with_stats(algo, model, calib_input, blockwise, captured=captured)

        left, right = (x_quant, x_float) if cfg['g_transposed'] else (x_float, x_quant)
        expected = {
            "H": _expected_covariance(x_quant, scale=scale),
            "G": _expected_covariance(left, right, scale),
            "R": _expected_covariance(x_float, scale=scale),}
        # The very first layer sees the calibration data on both passes, so G == H.
        expected_first = {
            "H": _expected_covariance(calib_input, scale=scale),
            "G": _expected_covariance(calib_input, scale=scale),}

        for layer_name, matrices in ((STATS_LAYER, expected), ("blocks.0.linear_1",
                                                               expected_first)):
            for matrix_name, matrix in matrices.items():
                for prefix in ("pre", "post"):
                    key = f"{layer_name}_{prefix}_{matrix_name}"
                    assert key in captured, key
                    torch.testing.assert_close(captured[key], matrix, msg=key)

    @pytest.mark.parametrize("algo", ["gptq", "qronos", "sopt"])
    @pytest.mark.parametrize("blockwise", [False, True])
    def test_layer_error_stats(self, algo, blockwise):
        """The reported per-layer errors must match a direct computation from the
        weights and the inputs."""
        cfg = _STATS_ALGO_CONFIG[algo]
        model, calib_input = _build_model_and_calibration_data()
        collector = _run_with_stats(algo, model, calib_input, blockwise)

        x_quant, x_float = self._target_layer_inputs(model, calib_input)
        layer = model.blocks[0].linear_2
        quant_weight = layer.quant_weight().value.to(torch.float32)
        weight = getattr(layer, cfg['weight_attr']).to(torch.float32)
        err = quant_weight - weight
        x_quant = x_quant.reshape(-1, STATS_HIDDEN_FEATURES).to(torch.float32)
        x_float = x_float.reshape(-1, STATS_HIDDEN_FEATURES).to(torch.float32)

        expected = {
            "rel_weight_err": (torch.norm(err, p='fro') / torch.norm(weight, p='fro')).item(),
            "rel_out_err":
                (torch.norm(x_quant @ err.T, p='fro') /
                 torch.norm(x_quant @ weight.T, p='fro')).item(),
            "fp_rel_out_err": (
                torch.norm(x_float @ weight.T - x_quant @ quant_weight.T, p='fro') /
                torch.norm(x_float @ weight.T, p='fro')).item(),}

        stats = collector.stats[STATS_LAYER]
        for prefix in ("pre", "post"):
            for key, value in expected.items():
                assert stats[f"{prefix}_{key}"] == pytest.approx(value, abs=1e-5), key

    @pytest.mark.parametrize("algo", ["gptq", "qronos", "sopt"])
    def test_block_error_stats(self, algo):
        """The block output error is measured against the fully-float reference,
        independently of the float cache the algorithm itself uses."""
        model, calib_input = _build_model_and_calibration_data()
        collector = _run_with_stats(algo, model, calib_input, blockwise=True)

        x_quant = x_float = calib_input
        for index, block in enumerate(model.blocks):
            with torch.no_grad():
                y_quant = block(x_quant)
                with _float_model(model):
                    y_float = block(x_float)
            expected = (torch.norm(y_quant - y_float, p='fro') /
                        torch.norm(y_float, p='fro')).item()
            stats = collector.stats["blocks"][f"blocks.{index}"]
            for prefix in ("pre", "post"):
                assert stats[f"{prefix}_block_rel_out_err"] == pytest.approx(expected, abs=1e-5)
            x_quant, x_float = y_quant, y_float

    def test_create_weight_orig_required(self):
        """GPxQ statistics need the float reference weights to be kept."""
        model, _ = _build_model_and_calibration_data()
        with collect_stats(DictStatsCollector()):
            with pytest.raises(ValueError, match="create_weight_orig"):
                gptq_mode(model, create_weight_orig=False)

    def test_statistics_are_opt_in(self):
        """Without a collector no covariance beyond H is allocated."""
        model, calib_input = _build_model_and_calibration_data()
        with torch.no_grad():
            with gptq_mode(model, create_weight_orig=True) as gptq:
                for _ in range(gptq.num_layers):
                    gptq.model(calib_input)
                    optimizers = [gptq.layers[n] for n in gptq.current_layer.layer_names]
                    assert all(not hasattr(o, 'R') for o in optimizers)
                    assert all(not o._stats_dual_pass for o in optimizers)
                    gptq.update()

    @pytest.mark.parametrize("blockwise", [False, True])
    def test_apply_gpxq_stats_does_not_modify_the_model(self, blockwise):
        """The standalone measurement reports the error of the model as-is."""
        model, calib_input = _build_model_and_calibration_data()
        reference = {n: p.detach().clone() for n, p in model.named_parameters()}

        with patch("brevitas_examples.llm.llm_quant.gpxq.send_to_device", lambda x, device: x):
            collector = apply_gpxq_stats(
                model, [{
                    "x": calib_input}], block_name="blocks" if blockwise else None)

        for name, param in model.named_parameters():
            torch.testing.assert_close(param, reference[name], msg=name)

        stats = collector.stats[STATS_LAYER]
        for key in ("rel_weight_err", "rel_out_err", "fp_rel_out_err"):
            assert stats[f"pre_{key}"] == pytest.approx(stats[f"post_{key}"], abs=1e-6)
            assert stats[f"pre_{key}"] > 0.0
