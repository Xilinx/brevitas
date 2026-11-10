# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for brevitas.graph.grid_optimization.core: how grid optimization obtains its
covariance matrices (the standalone hook that accumulates them, and the composition
with GPTQ and Qronos, i.e. which layers get a grid optimizer, the covariance
matrices GPTQ and Qronos hand to it and the column order used when
group-interleaved), and that its settings are orthogonal to the solver."""
from functools import partial

import pytest
import torch

from brevitas.graph.gptq import GPTQ
from brevitas.graph.grid_optimization import GridOptimizer
from brevitas.graph.grid_optimization import InterleavedGridOptimizer
from brevitas.graph.grid_optimization import StandaloneGridOptimizer
from brevitas.graph.grid_optimization.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.grid_optimization.solvers import ScaleSolverFamily
from brevitas.graph.grid_optimization.utils import compute_group_importance_permutation
from brevitas.graph.layerwise_hook import LayerHandler
from brevitas.graph.qronos import Qronos
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
import brevitas.nn as qnn
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.quant.mx_quant_ocp import MXInt8Weight
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.utils.torch_utils import StopFwdException

D = 8
GROUP_SIZE = 4
NUM_BLOCKS = 1


class _IntPerGroupWeight(MXInt8Weight):
    # Groupwise int weight quant with a parameter-from-stats scale (the scaling
    # grid optimization supports) instead of MX's power-of-two scale.
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    group_size = GROUP_SIZE


def _build_gptq(layer, **grid_optimization_kwargs):
    return GPTQ(
        layer,
        'layer',
        act_order=True,
        len_parallel_layers=1,
        create_weight_orig=False,
        num_blocks=NUM_BLOCKS,
        grid_optimizer_factory=partial(
            InterleavedGridOptimizer.for_gpxq, **grid_optimization_kwargs))


class _RecordingGridOptimizer:
    """Stands in for InterleavedGridOptimizer: keeps the defaults and records the handoff."""
    group_interleaved = False

    def resolve_blocksize(self, blocksize):
        return blocksize

    def on_error_correction_starts(self, H, G, perm):
        # Copies: the algorithm modifies its covariance matrices in place after the hook.
        self.H = H.clone()
        self.G = None if G is None else G.clone()
        self.perm = perm.clone()

    def on_column_reached(self, column_idx):
        pass

    def on_layer_finished(self):
        pass


def _random_covariance(generator):
    X = torch.randn(4 * D, D, generator=generator)
    return X.T @ X


def test_gptq_handoff_is_damped_h_in_natural_order():
    g = torch.Generator().manual_seed(0)
    H = _random_covariance(g)
    percdamp = .01
    recorder = _RecordingGridOptimizer()
    gptq = GPTQ(
        qnn.QuantLinear(D, 3, bias=False),
        'layer',
        act_order=True,
        len_parallel_layers=1,
        create_weight_orig=False,
        num_blocks=NUM_BLOCKS,
        grid_optimizer_factory=lambda layer: recorder)
    gptq.H = H.unsqueeze(0).clone()

    gptq.single_layer_update(percdamp=percdamp)

    assert recorder.G is None
    assert torch.equal(recorder.perm, torch.argsort(H.diag(), descending=True))
    expected_H = H + percdamp * H.diag().mean() * torch.eye(D)
    assert torch.allclose(recorder.H, expected_H)


def test_qronos_handoff_keeps_natural_order_and_transposes_g():
    g = torch.Generator().manual_seed(1)
    H = _random_covariance(g)
    G = torch.randn(D, D, generator=g)
    recorder = _RecordingGridOptimizer()
    qronos = Qronos(
        qnn.QuantLinear(D, 3, bias=False),
        'layer',
        act_order=True,
        len_parallel_layers=1,
        create_weight_orig=True,
        num_blocks=NUM_BLOCKS,
        grid_optimizer_factory=lambda layer: recorder)
    qronos.H, qronos.G = H.unsqueeze(0).clone(), G.unsqueeze(0).clone()

    qronos.single_layer_update()

    assert torch.equal(recorder.H, H)
    # The grid optimizer expects G = X_tilde^T X, the transpose of what Qronos
    # accumulates. Qronos adds its damping to the diagonal of G before the hook.
    off_diagonal = ~torch.eye(D, dtype=torch.bool)
    assert torch.equal(recorder.G[off_diagonal], G.T[off_diagonal])


@pytest.mark.parametrize("algorithm_class", [GPTQ, Qronos])
def test_act_order_is_group_contiguous_only_when_group_interleaved(algorithm_class):
    # Plain descending order interleaves the two groups, so the two orders differ.
    diag = torch.tensor([8., 1., 7., 2., 6., 3., 5., 4.5])
    H = torch.diag(diag)
    plain_order = torch.argsort(diag, descending=True)
    group_order, _ = compute_group_importance_permutation(diag, GROUP_SIZE)
    assert not torch.equal(plain_order, group_order)

    grid_optimizer = object.__new__(InterleavedGridOptimizer)
    grid_optimizer._unit_columns = lambda: GROUP_SIZE
    algo = object.__new__(algorithm_class)
    algo.grid_optimizer = grid_optimizer

    grid_optimizer.group_interleaved = True
    assert torch.equal(algo._act_order_permutation(H), group_order)

    grid_optimizer.group_interleaved = False
    assert torch.equal(algo._act_order_permutation(H), plain_order)

    algo.grid_optimizer = None
    assert torch.equal(algo._act_order_permutation(H), plain_order)


@pytest.mark.parametrize("layer_interleaved", [False, True])
def test_gptq_builds_grid_optimizer_for_supported_layer(layer_interleaved):
    layer = qnn.QuantLinear(
        D, 3, bias=False, weight_quant=_IntPerGroupWeight, weight_group_size=GROUP_SIZE)

    gptq = _build_gptq(layer, layer_interleaved=layer_interleaved, solver_device='cpu')

    assert isinstance(gptq.grid_optimizer, InterleavedGridOptimizer)
    assert gptq.grid_optimizer.group_interleaved is not layer_interleaved
    # Blocks are aligned with the quantization groups in both modes.
    assert gptq.blocksize == GROUP_SIZE


def test_gptq_skips_grid_optimization_for_unsupported_layer():
    # The default per-tensor weight quantizer does not store a parameter-from-stats scale.
    layer = qnn.QuantLinear(D, 3, bias=False)

    with pytest.warns(UserWarning, match="Skipping grid optimization"):
        gptq = _build_gptq(layer, solver_device='cpu')

    assert gptq.grid_optimizer is None
    assert gptq.blocksize == D // NUM_BLOCKS


def test_standalone_accumulates_h_and_g():
    g = torch.Generator().manual_seed(2)
    layer = qnn.QuantLinear(D, 3, bias=False)
    grid_optimizer = StandaloneGridOptimizer(
        layer,
        'layer',
        solves_mismatched_objective=True,
        matrix_dtype=torch.float64,
        solver_device='cpu')
    current_layer = LayerHandler()
    quant_inputs, float_inputs = [], []
    for batch_size in (3, 5):
        quant_inputs.append(torch.randn(batch_size, D, generator=g))
        float_inputs.append(torch.randn(batch_size, D, generator=g))
        # A quantized pass, then a float pass: the order tells them apart.
        with pytest.raises(StopFwdException):
            grid_optimizer.update_batch(layer, (quant_inputs[-1],), current_layer)
        with pytest.raises(StopFwdException):
            grid_optimizer.update_batch(layer, (float_inputs[-1],), current_layer)

    X_tilde = torch.cat(quant_inputs).double()
    X = torch.cat(float_inputs).double()
    N = X.shape[0]
    assert current_layer.layer_names == {'layer'}
    assert torch.allclose(grid_optimizer.H[0], 2 / N * X_tilde.T @ X_tilde)
    # G is in the layout the solvers expect, X_tilde^T X.
    assert torch.allclose(grid_optimizer.G[0], 2 / N * X_tilde.T @ X)


# -----------------------------------------------------------------------------
# The settings of grid optimization are orthogonal to the solver: every registered
# solver runs with every covariance mode it supports, either objective, either
# granularity and either schedule.
# -----------------------------------------------------------------------------
class _NarrowInt4PerChannelWeight(Int8WeightPerChannelFloat):
    """Signed 4-bit codes on a symmetric grid, so that no scale may be negative."""
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    bit_width = 4
    narrow_range = True
    scaling_min_val = 1e-10


class _NarrowInt4PerGroupWeight(_NarrowInt4PerChannelWeight):
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = GROUP_SIZE


def _optimized_scales(weight_quant, solver, covariance_mode, mismatched, greedy):
    torch.manual_seed(0)
    layer = qnn.QuantLinear(D, 3, bias=False, weight_quant=weight_quant)
    with torch.no_grad():
        layer(torch.randn(2, D))
    g = torch.Generator().manual_seed(4)
    H = _random_covariance(g)
    G = H + 0.1 * torch.randn(D, D, generator=g) if mismatched else None
    grid_optimizer = GridOptimizer(
        layer,
        solver_device='cpu',
        covariance_mode=covariance_mode,
        solver=solver,
        per_group_greedy=greedy)
    with torch.no_grad():
        grid_optimizer.update_grid(H, G)
    return layer.quant_weight().scale_.detach()


@pytest.mark.parametrize("solver", ['piso', 'chunked-piso'])
@pytest.mark.parametrize("covariance_mode", ['diagonal', 'identity'])
@pytest.mark.parametrize("mismatched", [False, True])
@pytest.mark.parametrize(
    "weight_quant", [_NarrowInt4PerChannelWeight, _NarrowInt4PerGroupWeight],
    ids=['per_channel', 'per_group'])
def test_every_setting_runs_with_every_solver(solver, covariance_mode, mismatched, weight_quant):
    scales = _optimized_scales(weight_quant, solver, covariance_mode, mismatched, False)
    assert torch.isfinite(scales).all() and (scales > 0).all()
    # Without a full covariance matrix, solving the groups one at a time changes nothing.
    greedy = _optimized_scales(weight_quant, solver, covariance_mode, mismatched, True)
    assert torch.allclose(greedy, scales)


def test_standalone_with_input_quantization_fits_the_quantized_inputs():
    g = torch.Generator().manual_seed(3)
    layer = qnn.QuantLinear(
        D,
        3,
        bias=False,
        weight_quant=_NarrowInt4PerChannelWeight,
        input_quant=Int8ActPerTensorFloat)
    x = torch.randn(5, D, generator=g)
    with torch.no_grad():
        layer(x)
    layer.eval()
    grid_optimizer = StandaloneGridOptimizer(
        layer, 'layer', matrix_dtype=torch.float64, solver_device='cpu')
    with pytest.raises(StopFwdException):
        grid_optimizer.update_batch(layer, (x,), LayerHandler())
    X_tilde = layer.input_quant(x).value.double()
    assert not torch.allclose(X_tilde, x.double())
    assert torch.allclose(grid_optimizer.H[0], 2 / 5 * X_tilde.T @ X_tilde)

    with torch.no_grad():
        grid_optimizer.single_layer_update()
    scales = layer.quant_weight().scale_.detach()
    assert torch.isfinite(scales).all() and (scales > 0).all()


def test_a_solver_without_dense_support_rejects_the_dense_mode():
    with pytest.raises(ValueError, match="only uses the diagonal"):
        GridOptimizer.validate_config('dense', 'piso')


class _RecordingDenseSolver(ScaleSolverFamily):
    """A solver registered outside grid optimization, which takes the diagonal blocks of
    the covariance matrix: it records what it is handed and returns the absmax scale."""
    supports_dense = True
    calls = []

    @classmethod
    def solve(cls, w, b, H, unscaled_grid, *, allow_negative_s, **kwargs):
        cls.calls.append((b.clone(), H.clone()))
        return w.abs().amax(-1) / unscaled_grid[-1]


SCALE_SOLVER_REGISTRY.register(names='test-recording-dense')(_RecordingDenseSolver)


def _record_dense_solver(per_group_greedy):
    _RecordingDenseSolver.calls.clear()
    torch.manual_seed(0)
    layer = qnn.QuantLinear(D, 3, bias=False, weight_quant=_NarrowInt4PerGroupWeight)
    with torch.no_grad():
        layer(torch.randn(2, D))
    g = torch.Generator().manual_seed(5)
    H = _random_covariance(g)
    G = torch.randn(D, D, generator=g)
    W = layer.weight.detach().clone()
    grid_optimizer = GridOptimizer(
        layer,
        solver_device='cpu',
        covariance_mode='dense',
        solver='test-recording-dense',
        per_group_greedy=per_group_greedy,
        per_group_greedy_reorder=False)
    with torch.no_grad():
        grid_optimizer.update_grid(H, G)
    return layer, W, H, G, _RecordingDenseSolver.calls


def test_a_dense_solver_solves_every_group_against_its_own_blocks():
    _, W, H, G, calls = _record_dense_solver(per_group_greedy=False)
    groups = [slice(0, GROUP_SIZE), slice(GROUP_SIZE, D)]
    [(b, H_blocks)] = calls
    assert torch.allclose(H_blocks, torch.stack([H[c, c] for c in groups]))
    # The independent-groups objective (PiSO paper, Eq. 11): b has no cross terms
    # between groups.
    assert torch.allclose(b, torch.stack([W[:, c] @ G[c, c].T for c in groups], dim=1))


def test_greedy_with_a_dense_solver_targets_the_groups_solved_so_far():
    layer, W, H, G, calls = _record_dense_solver(per_group_greedy=True)
    first, second = slice(0, GROUP_SIZE), slice(GROUP_SIZE, D)
    (b_first, H_first), (b_second, H_second) = calls
    assert torch.allclose(H_first, H[first, first].unsqueeze(0))
    assert torch.allclose(H_second, H[second, second].unsqueeze(0))
    # The sequential objective (PiSO paper, Eq. 12): the first group's target leaves out
    # the groups not solved yet, and the second group sees the reconstruction of the first.
    assert torch.allclose(b_first.reshape(3, -1), W[:, first] @ G[first, first].T)
    r_first = layer.quant_weight().value[:, first].detach()
    assert torch.allclose(b_second.reshape(3, -1), W @ G[second].T - r_first @ H[second, first].T)


class _Int4SignedScalePerChannelWeight(Int8WeightPerChannelFloat):
    """Signed 4-bit codes on the asymmetric grid [-8, 7], with a scale that may be negative."""
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS
    bit_width = 4
    narrow_range = False
    restrict_scaling_type = RestrictValueType.SIGNED_FP


@pytest.mark.parametrize("solver,valid", [('piso', True), ('chunked-piso', False)])
def test_only_solvers_of_negative_scales_accept_a_layer_that_allows_them(solver, valid):
    layer = qnn.QuantLinear(D, 3, bias=False, weight_quant=_Int4SignedScalePerChannelWeight)
    assert GridOptimizer.is_layer_valid(layer, solver) == valid
