# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""The llama.cpp searches run as grid optimization solvers.

What is checked here is that GridOptimizer, given H = 2 * diag(imatrix), writes
into a layer the scale (and zero-point) the corresponding kernel computes, for
per-group and per-channel, full and narrow range quantizers, including into the
nested quantizers of a K-format quantizer, which then export the steps the solver
chose; that the searches run interleaved with GPTQ; and that a solver skips the
layers it cannot fit.
"""

import warnings

import numpy as np
import pytest
import torch

from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.graph.gptq import GPTQ
from brevitas.graph.grid_optimization import GridOptimizer
from brevitas.graph.grid_optimization import InterleavedGridOptimizer
from brevitas.graph.grid_optimization.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.grid_optimization.utils import _build_grid
from brevitas.graph.grid_optimization.utils import _get_scale_quant_grid
from brevitas.graph.grid_optimization.utils import _get_zero_point_quant_grid
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.inject.enum import StatsOp
import brevitas.nn as qnn
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
# Importing the package registers the GGUF solvers.
import brevitas_examples.llm.gguf_export.search  # noqa: F401
from brevitas_examples.llm.gguf_export.search.kernels import affine_scale_search
from brevitas_examples.llm.gguf_export.search.kernels import symmetric_scale_search
from brevitas_examples.llm.gguf_export.search.objective import objective_weights

BLOCK = 32
N_IN = 128
N_OUT = 16


class SymmetricWeight(Int8WeightPerChannelFloat):
    """Signed 4-bit codes, one stored, sign-preserving scale per block."""
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    restrict_scaling_type = RestrictValueType.SIGNED_FP
    scaling_stats_op = StatsOp.SIGNED_MAX
    group_size = BLOCK
    bit_width = 4
    narrow_range = False
    scaling_min_val = 1e-10


class PositiveScaleWeight(SymmetricWeight):
    """Signed 4-bit codes with a scale that can only be positive."""
    restrict_scaling_type = RestrictValueType.FP
    scaling_stats_op = StatsOp.MAX


class AffineWeight(ShiftedUint8WeightPerChannelFloat):
    """Unsigned 4-bit codes, one stored scale and real-valued zero-point per block."""
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    zero_point_impl = ParameterFromStatsFromParameterZeroPoint
    restrict_scaling_type = RestrictValueType.FP
    group_size = BLOCK
    bit_width = 4
    narrow_range = False
    quantize_zero_point = False
    scaling_min_val = 1e-10


class RoundedZeroPointWeight(AffineWeight):
    quantize_zero_point = True


class PerChannelNarrowWeight(Int8WeightPerChannelFloat):
    """Signed 4-bit codes on a narrow grid [-7, 7], one stored scale per output channel."""
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    bit_width = 4
    narrow_range = True
    scaling_min_val = 1e-10


def build_layer(weight_quant, n_in=N_IN, seed=0):
    torch.manual_seed(seed)
    layer = qnn.QuantLinear(n_in, N_OUT, bias=False, weight_quant=weight_quant)
    # The scale and zero-point parameters only exist once the quantizer has run.
    with torch.no_grad():
        layer(torch.randn(2, n_in))
    return layer


def imatrix_for(n_in, seed=1):
    torch.manual_seed(seed)
    return torch.rand(n_in) + 0.1


def optimize(layer, solver, imatrix, covariance_mode='diagonal', **solver_kwargs):
    grid_optimizer = GridOptimizer(
        layer,
        solver_device='cpu',
        covariance_mode=covariance_mode,
        solver=solver,
        solver_kwargs=solver_kwargs)
    # H = 2/N X^T X, whose diagonal is twice llama.cpp's imatrix. A nested
    # quantizer has to read back the integers written.
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*neighbouring point')
        grid_optimizer.optimize_layer(torch.diag(2 * imatrix))


def fit_inputs(weight, imatrix, block=BLOCK):
    """The weights in blocks, and the weights of their fit, as the solver builds them:
    the diagonal of H = 2 * diag(imatrix), with the default magnitude."""
    blocked = weight.reshape(N_OUT, -1, block)
    h = (2 * imatrix).reshape(-1, block)
    quad, lin = objective_weights(blocked, h * blocked, h)
    return blocked, quad, lin


def export(layer):
    """What GGUF export reads off layer: the steps, and the decoded weights."""
    gguf_quants = pytest.importorskip("gguf.quants")
    from brevitas_examples.llm.gguf_export.export import _enable_gguf_export_caching
    from brevitas_examples.llm.gguf_export.export import _restore_gguf_export_caching
    from brevitas_examples.llm.gguf_export.quant import ggml_quant

    weight_quant = layer.weight_quant
    qtype = weight_quant.gguf_qtype
    layer.eval()
    prior = _enable_gguf_export_caching(layer)
    try:
        quant_weight = layer.quant_weight()
        block = ggml_quant(
            quant_weight.int().detach().cpu().numpy(),
            qtype,
            scale=weight_quant.scale,
            zero_point=weight_quant.zero_point,
            scale_of_scale=weight_quant.scale_of_scale,
            scale_of_zero_point=weight_quant.scale_of_zero_point)
        d = weight_quant.scale_of_scale
        dmin = weight_quant.scale_of_zero_point
    finally:
        _restore_gguf_export_caching(prior)
    decoded = gguf_quants.dequantize(block.reshape(N_OUT, -1), qtype).reshape(layer.weight.shape)
    return d, dmin, decoded, quant_weight.value.detach().numpy()


def test_the_sweep_writes_the_kernel_scale():
    layer = build_layer(SymmetricWeight)
    imatrix = imatrix_for(N_IN)
    blocked, quad, lin = fit_inputs(layer.weight.detach().clone(), imatrix)
    optimize(layer, 'gguf-sweep', imatrix)
    scale, _ = symmetric_scale_search(blocked, quad, lin, -8, 7)
    written = layer.quant_weight().scale_.reshape(scale.shape)
    assert torch.allclose(written, scale, rtol=1e-6, atol=0.)


def test_a_positive_scale_is_seeded_positive():
    layer = build_layer(PositiveScaleWeight)
    imatrix = imatrix_for(N_IN)
    blocked, quad, lin = fit_inputs(layer.weight.detach().clone(), imatrix)
    optimize(layer, 'gguf-sweep', imatrix)
    scale, _ = symmetric_scale_search(blocked, quad, lin, -8, 7, positive_scale=True)
    written = layer.quant_weight().scale_.reshape(scale.shape)
    assert (written > 0).all()
    assert torch.allclose(written, scale, rtol=1e-6, atol=0.)


def test_a_per_channel_narrow_range_layer_is_one_block_per_row():
    layer = build_layer(PerChannelNarrowWeight)
    imatrix = imatrix_for(N_IN)
    blocked, quad, lin = fit_inputs(layer.weight.detach().clone(), imatrix, block=N_IN)
    optimize(layer, 'gguf-sweep', imatrix)
    # A narrow grid is symmetric, so its scale is positive.
    scale, _ = symmetric_scale_search(blocked, quad, lin, -7, 7, positive_scale=True)
    written = layer.quant_weight().scale_.reshape(scale.shape)
    assert torch.allclose(written, scale, rtol=1e-6, atol=0.)


def test_the_sweep_fits_an_offset_for_a_quantizer_with_a_zero_point():
    layer = build_layer(AffineWeight)
    imatrix = imatrix_for(N_IN)
    blocked, quad, lin = fit_inputs(layer.weight.detach().clone(), imatrix)
    optimize(layer, 'gguf-sweep', imatrix)
    scale, offset = affine_scale_search(blocked, quad, lin, 15)
    quant_weight = layer.quant_weight()
    written = quant_weight.scale_.reshape(scale.shape)
    assert torch.allclose(written, scale, rtol=1e-6, atol=0.)
    written_offset = -quant_weight.zero_point_.reshape(scale.shape) * written
    assert torch.allclose(written_offset, offset, rtol=1e-5, atol=1e-7)


def solve(layer, solver, imatrix, **solver_kwargs):
    """What the solver returns for layer, without writing it."""
    weight = layer.weight.detach().clone()
    weight_quant = layer.weight_quant
    w = weight.reshape(N_OUT, -1, weight_quant.group_size)
    h = 2 * imatrix
    return SCALE_SOLVER_REGISTRY.get(solver).solve(
        w, (h * weight).reshape(w.shape),
        h,
        _build_grid(layer, torch.float32, 'cpu'),
        allow_negative_s=not weight_quant.restrict_scale_positive,
        with_offset=not weight_quant.is_signed,
        scale_quant_grid=_get_scale_quant_grid(layer),
        zero_point_quant_grid=_get_zero_point_quant_grid(layer),
        **solver_kwargs)


@pytest.mark.parametrize(
    "quantizer, solver, solver_kwargs",
    [
        # The configurations llama.cpp uses with an imatrix.
        (
            'GGUFQ4_KWeightQuant',
            'gguf-sweep',
            dict(magnitude_factor=2.0, magnitude_scope='shared_step')),
        # Q2_K's nested quantizers round its scales up and its offsets down.
        ('GGUFQ2_KWeightQuant', 'gguf-sweep', dict(magnitude_scope='shared_step')),
        ('GGUFQ6_KWeightQuant', 'gguf-sweep', dict(magnitude='one', scale_placement='seed')),],
    ids=['q4_k', 'q2_k', 'q6_k'])
def test_a_nested_grid_is_written_into_the_quantizer_and_exported(quantizer, solver, solver_kwargs):
    bq = pytest.importorskip("brevitas_examples.llm.gguf_export.base_quantizers")
    n_in = 512  # two super-blocks
    layer = build_layer(getattr(bq, quantizer), n_in=n_in)
    imatrix = imatrix_for(n_in)
    solution = solve(layer, solver, imatrix, **solver_kwargs)
    optimize(layer, solver, imatrix, **solver_kwargs)

    expected_scale = solution.scale.reshape(N_OUT, -1)
    live = expected_scale != 0
    quant_weight = layer.quant_weight()
    scale = quant_weight.scale_.reshape(expected_scale.shape)
    assert torch.allclose(scale[live], expected_scale[live], rtol=1e-5, atol=0.)
    exported_d, exported_dmin, decoded, reconstructed = export(layer)
    assert torch.allclose(
        exported_d.reshape(solution.scale_step.shape), solution.scale_step, rtol=1e-5, atol=0.)
    if solution.offset is not None:
        expected_offset = solution.offset.reshape(expected_scale.shape)
        offset = -quant_weight.zero_point_.reshape(expected_scale.shape) * scale
        assert torch.allclose(offset[live], expected_offset[live], rtol=1e-5, atol=1e-7)
        assert torch.allclose(
            exported_dmin.reshape(solution.offset_step.shape),
            solution.offset_step,
            rtol=1e-5,
            atol=0.)
    # The file holds the steps in half precision.
    np.testing.assert_allclose(
        decoded, reconstructed, rtol=0, atol=1e-2 * np.abs(reconstructed).max())


def test_batching_over_rows_does_not_change_the_result():
    imatrix = imatrix_for(N_IN)
    scales = []
    for solver_batch_size in (None, 3):
        layer = build_layer(AffineWeight)
        grid_optimizer = GridOptimizer(
            layer,
            solver_device='cpu',
            covariance_mode='diagonal',
            solver='gguf-sweep',
            solver_batch_size=solver_batch_size)
        with torch.no_grad():
            grid_optimizer.optimize_layer(torch.diag(2 * imatrix))
        quant_weight = layer.quant_weight()
        scales.append((quant_weight.scale_, quant_weight.zero_point_))
    assert torch.equal(scales[0][0], scales[1][0])
    assert torch.equal(scales[0][1], scales[1][1])


def test_identity_covariance_ignores_the_data():
    """With H = I every input channel weighs the same."""
    layer = build_layer(SymmetricWeight)
    blocked, quad, lin = fit_inputs(layer.weight.detach().clone(), torch.full((N_IN,), 0.5))
    optimize(layer, 'gguf-sweep', imatrix_for(N_IN), covariance_mode='identity')
    scale, _ = symmetric_scale_search(blocked, quad, lin, -8, 7)
    written = layer.quant_weight().scale_.reshape(scale.shape)
    assert torch.allclose(written, scale, rtol=1e-6, atol=0.)


@pytest.mark.parametrize(
    "solver, weight_quant, reason",
    [
        ('gguf-descent', AffineWeight, "fits no offset"),
        ('gguf-sweep', RoundedZeroPointWeight, "quantize_zero_point=False"),
        ('gguf-seed', RoundedZeroPointWeight, "quantize_zero_point=False"),],
    ids=['descent_with_offset', 'rounded_zp', 'seed'])
def test_a_layer_the_solver_cannot_fit_is_skipped(solver, weight_quant, reason):
    layer = build_layer(weight_quant)
    with pytest.warns(UserWarning, match=reason):
        assert not GridOptimizer.is_layer_valid(layer, solver)


def test_piso_skips_a_layer_with_a_nested_scale_grid():
    bq = pytest.importorskip("brevitas_examples.llm.gguf_export.base_quantizers")
    layer = build_layer(bq.GGUFQ6_KWeightQuant, n_in=512)
    with pytest.warns(UserWarning, match="grid of their own"):
        assert not GridOptimizer.is_layer_valid(layer, 'piso')


def test_a_nested_grid_runs_interleaved_with_gptq():
    bq = pytest.importorskip("brevitas_examples.llm.gguf_export.base_quantizers")
    n_in = 512
    layer = build_layer(bq.GGUFQ4_KWeightQuant, n_in=n_in)
    gptq = GPTQ(
        layer,
        'layer',
        act_order=True,
        len_parallel_layers=1,
        create_weight_orig=False,
        num_blocks=1,
        grid_optimizer_factory=lambda layer: InterleavedGridOptimizer.for_gpxq(
            layer, solver='gguf-sweep', solver_device='cpu'))
    assert gptq.grid_optimizer.group_interleaved
    # Error correction proceeds one run of groups sharing a scale step at a time.
    assert gptq.blocksize == 256
    torch.manual_seed(3)
    X = torch.randn(4 * n_in, n_in)
    gptq.H = (2 / X.shape[0] * X.T @ X).unsqueeze(0)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*neighbouring point')
        gptq.single_layer_update()
    assert torch.isfinite(layer.quant_weight().value).all()
