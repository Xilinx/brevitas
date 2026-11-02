# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for how grid optimization obtains its Hessians
(brevitas.graph.grid_optimization.core): the standalone hook that accumulates them,
and the hooks through which grid optimization interleaves with GPTQ and Qronos, i.e.
the Hessian handoff to the grid optimizer and the column order used with the
group-wise updater."""
from types import SimpleNamespace

import pytest
import torch

from brevitas.graph.grid_optimization import GridOptimizationGPTQ
from brevitas.graph.grid_optimization import GridOptimizationHandler
from brevitas.graph.grid_optimization import GridOptimizationQronos
from brevitas.graph.grid_optimization import GridOptimizer
from brevitas.graph.grid_optimization import GridOptimizerGroupInterleaved
from brevitas.graph.grid_optimization.utils import compute_group_importance_permutation
from brevitas.graph.layerwise_hook import LayerHandler
import brevitas.nn as qnn
from brevitas.utils.torch_utils import StopFwdException

D = 8
GROUP_SIZE = 4


def test_gptq_handoff_undoes_act_order_permutation():
    g = torch.Generator().manual_seed(0)
    H = torch.randn(D, D, generator=g)
    perm = torch.randperm(D, generator=g)
    gptq = object.__new__(GridOptimizationGPTQ)
    gptq.groups = 1
    # GPTQ permutes its Hessian in place before it calls the hook.
    gptq.H = H[perm][:, perm].unsqueeze(0)

    H_handoff, G_handoff = gptq._prepare_input_grid_optimizer(perm)

    assert G_handoff is None
    assert torch.equal(H_handoff, H)


def test_qronos_handoff_keeps_natural_order_and_transposes_g():
    g = torch.Generator().manual_seed(1)
    H = torch.randn(1, D, D, generator=g)
    G = torch.randn(1, D, D, generator=g)
    perm = torch.randperm(D, generator=g)
    qronos = object.__new__(GridOptimizationQronos)
    qronos.groups = 1
    # Qronos keeps H and G in the natural column order when it calls the hook.
    qronos.H, qronos.G = H, G

    H_handoff, G_handoff = qronos._prepare_input_grid_optimizer(perm)

    assert torch.equal(H_handoff, H[0])
    # The GridOptimizer expects G = X_tilde^T X, the transpose of what Qronos accumulates.
    assert torch.equal(G_handoff, G[0].T)


@pytest.mark.parametrize("grid_optimization_class", [GridOptimizationGPTQ, GridOptimizationQronos])
def test_act_order_is_group_contiguous_only_with_group_wise_updater(grid_optimization_class):
    # Plain descending order interleaves the two groups, so the two orders differ.
    diag = torch.tensor([8., 1., 7., 2., 6., 3., 5., 4.5])
    H = torch.diag(diag)
    plain_order = torch.argsort(diag, descending=True)
    group_order, _ = compute_group_importance_permutation(diag, GROUP_SIZE)
    assert not torch.equal(plain_order, group_order)

    algo = object.__new__(grid_optimization_class)
    algo.layer = SimpleNamespace(weight_quant=SimpleNamespace(group_size=GROUP_SIZE))

    algo.grid_optimizer = GridOptimizerGroupInterleaved(solver_device='cpu')
    assert torch.equal(algo._act_order_permutation(H), group_order)

    algo.grid_optimizer = GridOptimizer(solver_device='cpu')
    assert torch.equal(algo._act_order_permutation(H), plain_order)


def test_layer_handler_accumulates_h_and_g():
    g = torch.Generator().manual_seed(2)
    layer = qnn.QuantLinear(D, 3, bias=False)
    handler = GridOptimizationHandler(
        layer,
        'layer',
        len_parallel_layers=1,
        cross_act_objective=True,
        matrix_dtype=torch.float64,
        grid_optimizer=None)
    current_layer = LayerHandler()
    quant_inputs, float_inputs = [], []
    for batch_size in (3, 5):
        quant_inputs.append(torch.randn(batch_size, D, generator=g))
        float_inputs.append(torch.randn(batch_size, D, generator=g))
        # A quantized pass, then a float pass with the layer's weight quantization off.
        with pytest.raises(StopFwdException):
            handler.update_batch(layer, (quant_inputs[-1],), current_layer)
        layer.weight_quant.disable_quant = True
        with pytest.raises(StopFwdException):
            handler.update_batch(layer, (float_inputs[-1],), current_layer)
        layer.weight_quant.disable_quant = False

    X_tilde = torch.cat(quant_inputs).double()
    X = torch.cat(float_inputs).double()
    N = X.shape[0]
    assert current_layer.layer_names == {'layer'}
    assert torch.allclose(handler.H[0], 2 / N * X_tilde.T @ X_tilde)
    # G is in the layout the GridOptimizer expects, X_tilde^T X.
    assert torch.allclose(handler.G[0], 2 / N * X_tilde.T @ X)
