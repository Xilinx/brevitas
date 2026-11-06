# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for how grid optimization obtains its covariance matrices
(brevitas.graph.grid_optimization.core): the standalone hook that accumulates them,
and the composition with GPTQ and Qronos, i.e. which layers get a grid optimizer,
the covariance matrices GPTQ and Qronos hand to it and the column order used when
group-interleaved."""
from functools import partial
from types import SimpleNamespace

import pytest
import torch

from brevitas.graph.gptq import GPTQ
from brevitas.graph.grid_optimization import InterleavedGridOptimizer
from brevitas.graph.grid_optimization import StandaloneGridOptimizer
from brevitas.graph.grid_optimization.utils import compute_group_importance_permutation
from brevitas.graph.layerwise_hook import LayerHandler
from brevitas.graph.qronos import Qronos
from brevitas.inject.enum import ScalingImplType
import brevitas.nn as qnn
from brevitas.quant.mx_quant_ocp import MXInt8Weight
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
    grid_optimizer.layer = SimpleNamespace(weight_quant=SimpleNamespace(group_size=GROUP_SIZE))
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
        layer, 'layer', cross_act_objective=True, matrix_dtype=torch.float64, solver_device='cpu')
    # The covariance matrices are allocated by the first calibration batch.
    assert grid_optimizer.H is None and grid_optimizer.G is None
    current_layer = LayerHandler()
    quant_inputs, float_inputs = [], []
    for batch_size in (3, 5):
        quant_inputs.append(torch.randn(batch_size, D, generator=g))
        float_inputs.append(torch.randn(batch_size, D, generator=g))
        # A quantized pass, then a float pass with the layer's weight quantization off.
        with pytest.raises(StopFwdException):
            grid_optimizer.update_batch(layer, (quant_inputs[-1],), current_layer)
        layer.weight_quant.disable_quant = True
        with pytest.raises(StopFwdException):
            grid_optimizer.update_batch(layer, (float_inputs[-1],), current_layer)
        layer.weight_quant.disable_quant = False

    X_tilde = torch.cat(quant_inputs).double()
    X = torch.cat(float_inputs).double()
    N = X.shape[0]
    assert current_layer.layer_names == {'layer'}
    assert torch.allclose(grid_optimizer.H[0], 2 / N * X_tilde.T @ X_tilde)
    # G is in the layout the solvers expect, X_tilde^T X.
    assert torch.allclose(grid_optimizer.G[0], 2 / N * X_tilde.T @ X)
