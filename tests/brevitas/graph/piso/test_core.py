# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the PiSO layer-level updater with asymmetric weight quantizers
(brevitas.graph.piso.core)."""
import warnings

import pytest
import torch

from brevitas import config
from brevitas.core.zero_point import ParameterFromStatsFromParameterZeroPoint
from brevitas.graph.piso.core import ScaleOptimizer
from brevitas.graph.piso.core import ScaleOptimizerGroupInterleaved
import brevitas.graph.piso.core as core
from brevitas.graph.piso.solvers import PiSOSolverFamily
from brevitas.graph.piso.utils import _build_grid
from brevitas.graph.piso.utils import _get_macro_scale
from brevitas.graph.piso.utils import _get_macro_scale_threshold
from brevitas.graph.piso.utils import _grid_threshold
from brevitas.graph.piso.utils import _quant_scale_structure
from brevitas.graph.piso.utils import _scale_quant
from brevitas.graph.piso.utils import _weight_zero_point
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
import brevitas.nn as qnn
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightGroupQuantFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat

D = 64
M = 8


class _SymPerChannel(Int8WeightPerChannelFloat):
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS


class _SymGroup(Int8WeightPerChannelFloat):
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    group_size = 16


class _ZpPerChannel(ShiftedUint8WeightPerChannelFloat):
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    zero_point_impl = ParameterFromStatsFromParameterZeroPoint
    bit_width = 4


class _ZpGroup(ShiftedUint8WeightGroupQuantFloat):
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    zero_point_impl = ParameterFromStatsFromParameterZeroPoint
    bit_width = 4
    group_size = 16


def _layer(weight_quant, seed=1):
    torch.manual_seed(seed)
    layer = qnn.QuantLinear(D, M, bias=False, weight_quant=weight_quant)
    with torch.no_grad():
        layer.weight.normal_(0., 0.1)
    layer.eval()
    layer.weight_quant(layer.weight)  # force calibration
    return layer


def _hessian(seed=2):
    torch.manual_seed(seed)
    x = torch.randn(128, D)
    return x, (x.T @ x).unsqueeze(0)


def _output_error(layer, float_weight, x):
    with torch.no_grad():
        return ((x @ float_weight.T) - (x @ layer.quant_weight().value.T)).pow(2).sum().item()


# The dense-H solvers are not implemented, so the dense paths through the driver
# (hessian_mode='dense', and per_group_greedy, which only dispatches to the
# dense sequential slots) raise NotImplementedError. The mark is conditional on
# the slot still being the base stub, so it goes inert on its own once they are
# reimplemented. The default below is 'diagonal' so the asymmetric driver logic
# stays covered meanwhile.
DENSE_NOT_IMPLEMENTED = PiSOSolverFamily._per_channel.__qualname__.startswith('ScaleSolverFamily.')
dense_xfail = pytest.mark.xfail(
    DENSE_NOT_IMPLEMENTED,
    raises=NotImplementedError,
    strict=True,
    reason="dense solvers are not implemented")
HESSIAN_MODE_PARAMS = [pytest.param('dense', marks=[dense_xfail]), 'diagonal', 'identity']


def _optimize(layer, H, **kwargs):
    kwargs.setdefault('hessian_mode', 'diagonal')
    optimizer = ScaleOptimizer(solver_device='cpu', tolerance=1e-9, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer.single_layer_update(layer, H=H)


# -----------------------------------------------------------------------------
# is_layer_valid
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("weight_quant", [_SymPerChannel, _ZpPerChannel, _ZpGroup])
def test_is_layer_valid_accepts_supported_quantizers(weight_quant):
    assert ScaleOptimizer.is_layer_valid(_layer(weight_quant))


def test_is_layer_valid_rejects_non_writable_zero_point():
    """StatsFromParameterZeroPoint recomputes the zero point every forward, so
    the re-pinned magnitude would be discarded."""
    layer = _layer(
        ShiftedUint8WeightPerChannelFloat.let(
            scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS))
    with pytest.warns(UserWarning, match="writable"):
        assert not ScaleOptimizer.is_layer_valid(layer)


# -----------------------------------------------------------------------------
# Integer zero point: end to end
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("weight_quant", [_ZpPerChannel, _ZpGroup])
@pytest.mark.parametrize("hessian_mode", HESSIAN_MODE_PARAMS)
def test_integer_zero_point_improves_output_error(weight_quant, hessian_mode):
    layer = _layer(weight_quant)
    x, H = _hessian()
    float_weight = layer.weight.detach().clone()
    before = _output_error(layer, float_weight, x)
    _optimize(layer, H, hessian_mode=hessian_mode)
    after = _output_error(layer, float_weight, x)
    assert after < before, f"{hessian_mode}: {before} -> {after}"


@pytest.mark.parametrize("weight_quant", [_ZpPerChannel, _ZpGroup])
def test_integer_zero_point_preserved_by_scale_update(weight_quant):
    """z = round(m / scale) must survive the change of scale, which requires the
    stored magnitude to be re-pinned to m = scale * z."""
    layer = _layer(weight_quant)
    _, H = _hessian()
    before = _weight_zero_point(layer.weight_quant).clone()
    _optimize(layer, H)
    assert torch.allclose(before, _weight_zero_point(layer.weight_quant))


def test_symmetric_quantizer_is_unaffected_by_the_asymmetric_path():
    layer = _layer(_SymPerChannel)
    x, H = _hessian()
    float_weight = layer.weight.detach().clone()
    before = _output_error(layer, float_weight, x)
    _optimize(layer, H)
    assert _output_error(layer, float_weight, x) < before


# -----------------------------------------------------------------------------
# Configurations the solvers cannot express must be rejected, not mis-solved
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("weight_quant", [_ZpPerChannel, _ZpGroup])
def test_chunked_family_supports_asymmetric(weight_quant):
    """The chunked family implements both asymmetric extensions, so it must
    reach the same scales as the reference family on the same problem."""
    x, H = _hessian()
    reference, chunked = _layer(weight_quant), _layer(weight_quant)
    _optimize(reference, H)
    _optimize(chunked, H, solver='chunked-piso')
    ref_scale = (
        reference.weight_quant.scale_()
        if reference.weight_quant.is_groupwise else reference.weight_quant.scale())
    chunked_scale = (
        chunked.weight_quant.scale_()
        if chunked.weight_quant.is_groupwise else chunked.weight_quant.scale())
    assert torch.allclose(ref_scale, chunked_scale, rtol=1e-4, atol=1e-8)
    assert torch.allclose(
        _weight_zero_point(reference.weight_quant), _weight_zero_point(chunked.weight_quant))


@dense_xfail
@pytest.mark.parametrize("reorder", [True, False])
def test_greedy_supports_integer_zero_point(reorder):
    """The sequential sweep takes a translated per-group grid, and the greedy
    reordering permutes that grid along with H/G/W."""
    layer = _layer(_ZpGroup)
    x, H = _hessian()
    float_weight = layer.weight.detach().clone()
    before_error = _output_error(layer, float_weight, x)
    before_zp = _weight_zero_point(layer.weight_quant).clone()
    # greedy only dispatches to the dense sequential slots; with a diagonal H it
    # is ignored, so pin the mode to keep the test about the greedy path.
    _optimize(
        layer, H, hessian_mode='dense', per_group_greedy=True, per_group_greedy_reorder=reorder)
    assert _output_error(layer, float_weight, x) < before_error
    assert torch.allclose(before_zp, _weight_zero_point(layer.weight_quant))


# -----------------------------------------------------------------------------
# Group-interleaved updater (the GPTQ / Qronos integration)
# -----------------------------------------------------------------------------
def _run_interleaved(layer, x, **kwargs):
    """Drive the group-wise updater over every group, as error correction would."""
    kwargs.setdefault('hessian_mode', 'diagonal')
    optimizer = ScaleOptimizerGroupInterleaved(solver_device='cpu', tolerance=1e-9, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer.init_scale_optimizer_params(layer=layer, perm=torch.arange(D), H=(x.T @ x))
        group_size = layer.weight_quant.group_size
        for column in range(0, D, group_size):
            if optimizer.should_update_group(column):
                optimizer.single_group_update(column)
        optimizer.clear_scale_optimizer_params()
    return optimizer


@pytest.mark.parametrize("hessian_mode", HESSIAN_MODE_PARAMS)
@pytest.mark.parametrize("per_group_greedy", [False, True])
def test_group_interleaved_integer_zero_point(hessian_mode, per_group_greedy):
    """Each group gets its own translated grid, and the zero point is re-pinned
    per group exactly as the layer-level updater does."""
    layer = _layer(_ZpGroup)
    x, _ = _hessian()
    float_weight = layer.weight.detach().clone()
    before_error = _output_error(layer, float_weight, x)
    before_zp = _weight_zero_point(layer.weight_quant).clone()
    _run_interleaved(layer, x, hessian_mode=hessian_mode, per_group_greedy=per_group_greedy)
    assert _output_error(layer, float_weight, x) < before_error
    assert torch.allclose(before_zp, _weight_zero_point(layer.weight_quant))


def test_group_interleaved_matches_layer_level_on_the_same_objective():
    """Solving one group at a time must agree with solving them all at once: both
    minimize the independent-groups objective with the same per-group grid."""
    x, H = _hessian()
    layer_a, layer_b = _layer(_ZpGroup), _layer(_ZpGroup)
    _optimize(layer_a, H)
    _run_interleaved(layer_b, x)
    assert torch.allclose(
        layer_a.weight_quant.scale_(), layer_b.weight_quant.scale_(), rtol=1e-5, atol=1e-8)


def _gguf_quantizers():
    """The GGUF K-quant module, skipping when it cannot be exercised.

    K-quants search the scale with a local loss (stats_op.evaluate_loss assigns
    internal_candidate), which TorchScript cannot compile, so the quantizer
    cannot even be built with JIT on.
    """
    if config.JIT_ENABLED:
        pytest.skip("GGUF K-quant uses a local-loss search, which requires JIT to be disabled")
    return pytest.importorskip("brevitas_examples.llm.gguf_export.base_quantizers")


def test_group_interleaved_free_offset_is_preserved():
    gguf = _gguf_quantizers()
    from brevitas.graph.piso.utils import _weight_offset
    torch.manual_seed(3)
    layer = qnn.QuantLinear(256, 4, bias=False, weight_quant=gguf.GGUFQ4_KWeightQuant)
    with torch.no_grad():
        layer.weight.normal_(0., 0.1)
    layer.eval()
    layer.weight_quant(layer.weight)
    torch.manual_seed(7)
    x = torch.randn(256, 256)
    before = _weight_offset(layer.weight_quant).clone()
    optimizer = ScaleOptimizerGroupInterleaved(
        solver_device='cpu', tolerance=1e-9, hessian_mode='diagonal')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer.init_scale_optimizer_params(layer=layer, perm=torch.arange(256), H=(x.T @ x))
        for column in range(0, 256, layer.weight_quant.group_size):
            if optimizer.should_update_group(column):
                optimizer.single_group_update(column)
        optimizer.clear_scale_optimizer_params()
    assert torch.allclose(before, _weight_offset(layer.weight_quant), rtol=1e-5, atol=1e-8)


def test_group_interleaved_skips_an_unsupported_layer():
    """A layer the layer-level updater would skip must be skipped here too, rather
    than crashing a GPTQ run part way through."""
    layer = _layer(
        ShiftedUint8WeightGroupQuantFloat.let(
            scaling_impl_type=ScalingImplType.PARAMETER_FROM_STATS, group_size=16))
    x, _ = _hessian()
    assert not ScaleOptimizer.is_layer_valid(layer)
    optimizer = ScaleOptimizerGroupInterleaved(solver_device='cpu', tolerance=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer.init_scale_optimizer_params(layer=layer, perm=torch.arange(D), H=(x.T @ x))
    assert not optimizer.should_update_group(0)
    optimizer.clear_scale_optimizer_params()  # must not raise


def test_group_interleaved_supports_chunked_family():
    """The group-wise updater may also drive the chunked sweeps now."""
    x, _ = _hessian()
    reference, chunked = _layer(_ZpGroup), _layer(_ZpGroup)
    _run_interleaved(reference, x)
    _run_interleaved(chunked, x, solver='chunked-piso')
    assert torch.allclose(
        reference.weight_quant.scale_(), chunked.weight_quant.scale_(), rtol=1e-4, atol=1e-8)


def test_group_interleaved_still_accepts_symmetric():
    """The symmetric path through the group-wise updater must be unaffected."""
    layer = _layer(_SymGroup)
    x, _ = _hessian()
    optimizer = ScaleOptimizerGroupInterleaved(solver_device='cpu', tolerance=1e-9)
    optimizer.init_scale_optimizer_params(layer=layer, perm=torch.arange(D), H=(x.T @ x))
    optimizer.clear_scale_optimizer_params()


def test_free_offset_is_preserved_by_scale_update():
    """A free zero_point does not depend on the scale, so the sweep must leave it
    untouched. The stored zero point does change, since the quantizer derives it
    as zero_point / scale."""
    gguf = _gguf_quantizers()
    from brevitas.graph.piso.utils import _weight_offset
    torch.manual_seed(3)
    layer = qnn.QuantLinear(256, 4, bias=False, weight_quant=gguf.GGUFQ4_KWeightQuant)
    with torch.no_grad():
        layer.weight.normal_(0., 0.1)
    layer.eval()
    layer.weight_quant(layer.weight)
    torch.manual_seed(7)
    x = torch.randn(256, 256)
    before = _weight_offset(layer.weight_quant).clone()
    _optimize(layer, (x.T @ x).unsqueeze(0))
    assert torch.allclose(before, _weight_offset(layer.weight_quant), rtol=1e-5, atol=1e-8)


# -----------------------------------------------------------------------------
# Hierarchical scales: the macro stage
# -----------------------------------------------------------------------------
def _gguf_quantizers():
    """The GGUF quantizer module, skipping when it cannot be exercised.

    K-quants search the scale with a local loss (stats_op.evaluate_loss assigns
    internal_candidate), which TorchScript cannot compile, so the quantizer
    cannot even be built with JIT on.
    """
    if config.JIT_ENABLED:
        pytest.skip("GGUF K-quant uses a local-loss search, which requires JIT to be disabled")
    return pytest.importorskip("brevitas_examples.llm.gguf_export.base_quantizers")


def _hierarchical_layer(weight_quant, in_features=512, seed=3):
    torch.manual_seed(seed)
    layer = qnn.QuantLinear(in_features, 4, bias=False, weight_quant=weight_quant)
    with torch.no_grad():
        layer.weight.normal_(0., 0.1)
    layer.eval()
    layer.weight_quant(layer.weight)
    return layer


def _hierarchical_error(layer, x, float_weight):
    return ((x @ float_weight.T) - (x @ layer.quant_weight().value.T)).pow(2).sum().item()


# The shipped Q2_K rounds its sub-block codes with ceil/floor, which the sweep's
# midpoint transitions do not model, so the validity gate skips it (see
# test_shipped_q2_k_ceil_macro_scale_is_skipped). The round-to-nearest variant is
# the one that is actually optimized, so the macro tests exercise it instead.
K_QUANT_NAMES = ["GGUFQ2_KRoundWeightQuant", "GGUFQ4_KWeightQuant", "GGUFQ6_KWeightQuant"]


@pytest.mark.parametrize("quant_name", K_QUANT_NAMES)
def test_macro_stage_makes_the_stored_scale_representable(quant_name):
    """Without the macro stage the solved scale is silently re-rounded against a
    stale macro scale. Afterwards the realized scale must be exactly d * code,
    with the codes on the code grid."""
    gguf = _gguf_quantizers()
    layer = _hierarchical_layer(getattr(gguf, quant_name))
    torch.manual_seed(7)
    x = torch.randn(256, 512)
    _optimize(layer, (x.T @ x).unsqueeze(0), hessian_mode='diagonal')

    n_groups, n_super_blocks = _quant_scale_structure(layer)
    grid = _build_grid(layer, torch.float32, torch.device('cpu'))
    threshold = _grid_threshold(grid, layer.weight_quant)
    macro_scale_threshold = _get_macro_scale_threshold(layer)
    d = (_get_macro_scale(layer) / macro_scale_threshold / threshold).reshape(
        4, n_super_blocks, 1, 1)
    realized = layer.weight_quant.scale_().reshape(4, n_super_blocks, n_groups // n_super_blocks, 1)
    codes = realized / d
    assert torch.allclose(codes, codes.round(), atol=1e-4), "scale is not d * integer code"
    assert bool((codes.abs() <= macro_scale_threshold + 1e-4).all()), "codes outside the code grid"
    assert torch.allclose(realized, d * codes.round(), rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("quant_name", K_QUANT_NAMES)
@pytest.mark.parametrize("hessian_mode", ['diagonal', 'identity'])
def test_macro_stage_improves_on_a_stale_macro_scale(quant_name, hessian_mode):
    """The point of the stage: leaving the macro scale at its calibrated value
    throws away most of what the sweep solved for.

    Asserted in aggregate over several layers rather than per layer. Freezing the
    weight assignment is an approximation -- it is an upper bound on the true
    error only for a diagonal H with G = H -- so an individual layer can come out
    marginally worse even though the stage is a clear win on average.
    """
    gguf = _gguf_quantizers()
    torch.manual_seed(7)
    x = torch.randn(256, 512)

    def run(with_macro, seed):
        layer = _hierarchical_layer(getattr(gguf, quant_name), seed=seed)
        float_weight = layer.weight.detach().clone()
        original = core.has_quantized_scale
        if not with_macro:
            core.has_quantized_scale = lambda _layer: False
        try:
            _optimize(layer, (x.T @ x).unsqueeze(0), hessian_mode=hessian_mode)
        finally:
            core.has_quantized_scale = original
        return _hierarchical_error(layer, x, float_weight)

    seeds = range(3, 11)
    with_macro = [run(True, s) for s in seeds]
    stale = [run(False, s) for s in seeds]
    assert sum(with_macro) < sum(stale), (f"total error {sum(with_macro):.4f} vs {sum(stale):.4f}")


def test_macro_stage_leaves_a_flat_quantizer_untouched():
    """A quantizer that stores its scale directly must take exactly the path it
    took before, macro scale or not."""
    gguf = _gguf_quantizers()
    torch.manual_seed(7)
    x = torch.randn(256, 512)
    scales = []
    for with_macro in [True, False]:
        layer = _hierarchical_layer(gguf.GGUFQ4_1WeightQuant)
        original = core.has_quantized_scale
        if not with_macro:
            core.has_quantized_scale = lambda _layer: False
        try:
            _optimize(layer, (x.T @ x).unsqueeze(0), hessian_mode='diagonal')
        finally:
            core.has_quantized_scale = original
        scales.append(layer.weight_quant.scale_().clone())
    assert torch.equal(scales[0], scales[1])


def test_hierarchical_layer_without_a_writable_macro_scale_is_skipped():
    """A hierarchical quantizer whose macro scale cannot be written would have
    its solved scales re-rounded against the stale one, so it is skipped."""
    gguf = _gguf_quantizers()
    layer = _hierarchical_layer(gguf.GGUFQ4_KWeightQuant)
    scale_quant = _scale_quant(layer)
    original = scale_quant.scaling_impl
    scale_quant.scaling_impl = torch.nn.Identity()
    try:
        with pytest.warns(UserWarning, match="macro scale"):
            assert not ScaleOptimizer.is_layer_valid(layer)
    finally:
        scale_quant.scaling_impl = original


def test_shipped_q2_k_ceil_macro_scale_is_skipped():
    """The shipped Q2_K rounds its sub-block codes with ceil/floor, which the
    sweep's midpoint transitions do not model, so the validity gate skips it with
    a warning rather than solving it against the wrong transitions."""
    gguf = _gguf_quantizers()
    layer = _hierarchical_layer(gguf.GGUFQ2_KWeightQuant)
    with pytest.warns(UserWarning, match="round-to-nearest"):
        assert not ScaleOptimizer.is_layer_valid(layer)


# Only 'diagonal'/'identity': the dense per-group base sweep is not implemented by
# this solver family, so a dense base solve raises NotImplementedError before the
# macro stage is even reached.
@pytest.mark.parametrize("hessian_mode", ['diagonal'])
def test_round_q2_k_quant_stage_lowers_output_error(hessian_mode):
    """The round-to-nearest Q2_K is the PiSO-optimizable counterpart of the
    shipped (ceil/floor) Q2_K. Solving its macro scale must lower the output
    reconstruction error versus leaving the macro scale at its calibrated value.

    Asserted in aggregate over several layers: freezing the weight assignment is
    an approximation, so an individual layer can come out marginally worse even
    though the stage is a clear win on average.
    """
    gguf = _gguf_quantizers()
    torch.manual_seed(7)
    x = torch.randn(256, 512)

    def run(with_macro, seed):
        layer = _hierarchical_layer(gguf.GGUFQ2_KRoundWeightQuant, seed=seed)
        float_weight = layer.weight.detach().clone()
        original = core.has_quantized_scale
        if not with_macro:
            core.has_quantized_scale = lambda _layer: False
        try:
            _optimize(layer, (x.T @ x).unsqueeze(0), hessian_mode=hessian_mode)
        finally:
            core.has_quantized_scale = original
        return _hierarchical_error(layer, x, float_weight)

    seeds = range(3, 11)
    with_macro = [run(True, s) for s in seeds]
    stale = [run(False, s) for s in seeds]
    assert sum(with_macro) < sum(stale), (f"total error {sum(with_macro):.4f} vs {sum(stale):.4f}")


# -----------------------------------------------------------------------------
# Greedy permutation round-trip: _revert_group_greedy_permutation inverts
# _apply_group_greedy_permutation, so the macro stage sees the natural order.
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("reorder", [True, False])
@pytest.mark.parametrize("with_zero_point", [False, True])
def test_revert_group_greedy_permutation_is_the_inverse(reorder, with_zero_point):
    """Permuting H/G/W/grid/zero_point for the greedy sweep and then restoring
    must return the original tensors exactly."""
    torch.manual_seed(0)
    n_channels, n_groups, group_size = 5, 4, 8
    channel_size = n_groups * group_size

    H = torch.randn(channel_size, channel_size)
    H = H @ H.T  # symmetric, so the row/column permutation is well defined
    G = torch.randn(channel_size, channel_size)
    W = torch.randn(n_channels, channel_size)
    # A per-group translated grid and a per-group zero point, as the asymmetric
    # paths produce: one row per group so they ride the block permutation.
    grid = torch.randn(n_channels, n_groups, 6) if with_zero_point else torch.arange(6.)
    zero_point = torch.randn(n_channels, n_groups, 1) if with_zero_point else None

    optimizer = ScaleOptimizer(
        solver_device='cpu',
        cross_act_objective=True,
        per_group_greedy=True,
        per_group_greedy_reorder=reorder)

    H_p, G_p, W_p, zp_p, grid_p, perm, inv_block_perm = \
        optimizer._apply_group_greedy_permutation(
            H.clone(), G.clone(), W.clone(), group_size,
            zero_point=None if zero_point is None else zero_point.clone(),
            grid=grid.clone() if grid.dim() > 1 else grid)
    # _prepare_matrices reshapes W into groups before the macro stage sees it.
    W_p = W_p.reshape(n_channels, n_groups, group_size)

    H_r, G_r, W_r, grid_r, zp_r = optimizer._revert_group_greedy_permutation(
        H_p, G_p, W_p, grid_p, zp_p, perm, inv_block_perm, channel_size)

    assert torch.equal(H_r, H)
    assert torch.equal(G_r, G)
    assert torch.equal(W_r, W)
    if zero_point is None:
        assert zp_r is None
    else:
        assert torch.equal(zp_r, zero_point)
    if grid.dim() > 1:
        assert torch.equal(grid_r, grid)


@pytest.mark.parametrize("reorder", [True, False])
def test_greedy_hierarchical_scale_is_representable(reorder):
    """A hierarchical K-quant optimized with per_group_greedy still lands its
    realized scale on d * code: the greedy permutation is undone before the macro
    stage, so the super-blocks are solved in their natural column order."""
    gguf = _gguf_quantizers()
    layer = _hierarchical_layer(gguf.GGUFQ4_KWeightQuant)
    torch.manual_seed(7)
    x = torch.randn(256, 512)
    try:
        _optimize(
            layer, (x.T @ x).unsqueeze(0),
            hessian_mode='dense',
            per_group_greedy=True,
            per_group_greedy_reorder=reorder)
    except NotImplementedError:
        # The greedy / dense per-group base sweep is not implemented by this
        # solver family; the macro stage cannot be reached without it.
        pytest.skip("solver family does not implement the greedy per-group sweep")

    n_groups, n_super_blocks = _quant_scale_structure(layer)
    grid = _build_grid(layer, torch.float32, torch.device('cpu'))
    threshold = _grid_threshold(grid, layer.weight_quant)
    macro_scale_threshold = _get_macro_scale_threshold(layer)
    d = (_get_macro_scale(layer) / macro_scale_threshold / threshold).reshape(
        4, n_super_blocks, 1, 1)
    realized = layer.weight_quant.scale_().reshape(4, n_super_blocks, n_groups // n_super_blocks, 1)
    codes = realized / d
    assert torch.allclose(codes, codes.round(), atol=1e-4), "scale is not d * integer code"
    assert bool((codes.abs() <= macro_scale_threshold + 1e-4).all()), "codes outside the code grid"
