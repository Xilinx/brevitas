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
from brevitas.graph.piso.solvers import PiSOSolverFamily
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
