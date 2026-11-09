# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""What a fit is weighed by."""

import pytest
import torch

from brevitas_examples.llm.gguf_export.search.objective import MAGNITUDE_FORMS
from brevitas_examples.llm.gguf_export.search.objective import objective_weights

N_OUT, N_RUNS, N_SUB, BLOCK = 4, 2, 8, 32


def _weights(seed=0):
    torch.manual_seed(seed)
    x = torch.randn(N_OUT, N_RUNS, N_SUB, BLOCK)
    h = torch.rand(N_RUNS, N_SUB, BLOCK) + 0.1
    return x, h


@pytest.mark.parametrize("factor", [1.0, 2.0])
def test_rms_quad_over_a_shared_step(factor):
    x, h = _weights()
    mean_sq = (x * x).reshape(N_OUT, N_RUNS, -1).mean(-1).reshape(N_OUT, N_RUNS, 1, 1)
    term = torch.sqrt(factor * mean_sq + x * x)
    quad, lin = objective_weights(x, h * x, h, 'rms_quad', factor, 'shared_step')
    assert torch.allclose(quad, h * term)
    assert torch.allclose(lin, h * x * term)


def test_rms_quad_over_a_row():
    x, h = _weights()
    flat, flat_h = x.reshape(N_OUT, N_RUNS * N_SUB, BLOCK), h.reshape(N_RUNS * N_SUB, BLOCK)
    mean_sq = (x * x).reshape(N_OUT, -1).mean(-1).reshape(N_OUT, 1, 1)
    quad, _ = objective_weights(flat, flat_h * flat, flat_h, 'rms_quad', 1.0, 'row')
    assert torch.allclose(quad, flat_h * torch.sqrt(mean_sq + flat * flat))


def test_rms_plus_abs_over_a_block():
    x, h = _weights()
    quad, _ = objective_weights(x, h * x, h, 'rms_plus_abs', scope='block')
    assert torch.allclose(quad, h * (torch.sqrt((x * x).mean(-1, keepdim=True)) + x.abs()))


def test_one_leaves_the_covariance_alone():
    x, h = _weights()
    b = torch.randn_like(x)
    quad, lin = objective_weights(x, b, h, 'one')
    assert torch.equal(quad, h.expand_as(x))
    assert torch.equal(lin, b)


def test_the_scopes_disagree():
    x, h = _weights()
    seen = [objective_weights(x, h * x, h, scope=s)[0] for s in ('block', 'shared_step', 'row')]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert not torch.allclose(seen[a], seen[b])


def test_a_shared_step_scope_needs_runs_of_groups():
    x, h = _weights()
    flat = x.reshape(N_OUT, N_RUNS * N_SUB, BLOCK)
    with pytest.raises(ValueError, match="shared_step"):
        objective_weights(flat, flat, h.reshape(-1, BLOCK), scope='shared_step')


@pytest.mark.parametrize("form", MAGNITUDE_FORMS)
def test_every_form_is_positive_on_nonzero_weights(form):
    x, h = _weights()
    quad, _ = objective_weights(x, h * x, h, form)
    assert (quad > 0).all()
