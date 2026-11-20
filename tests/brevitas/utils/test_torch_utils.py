# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch.nn as nn

from brevitas.utils.torch_utils import materialized_params


def _offloaded_linear(calls, name):
    # Stand-in for a module offloaded by offload_model, which attaches these two functions
    module = nn.Linear(4, 4)
    module.allocate_params = lambda m: calls.append(('allocate', name))
    module.offload_params = lambda m: calls.append(('offload', name))
    return module


def test_materialized_params_not_offloaded_is_noop():
    module = nn.Linear(4, 4)
    with materialized_params(module):
        pass
    assert not hasattr(module, 'allocate_params')


def test_materialized_params_allocates_and_offloads():
    calls = []
    module = _offloaded_linear(calls, 'a')
    with materialized_params(module):
        assert calls == [('allocate', 'a')]
    assert calls == [('allocate', 'a'), ('offload', 'a')]


def test_materialized_params_is_reentrant():
    # A nested block must not reload the weights (it would discard what the outer block wrote)
    calls = []
    module = _offloaded_linear(calls, 'a')
    with materialized_params(module):
        with materialized_params(module):
            pass
        assert calls == [('allocate', 'a')]
    assert calls == [('allocate', 'a'), ('offload', 'a')]


def test_materialized_params_offloads_on_exception():
    calls = []
    module = _offloaded_linear(calls, 'a')
    with pytest.raises(RuntimeError):
        with materialized_params(module):
            raise RuntimeError
    assert calls == [('allocate', 'a'), ('offload', 'a')]
    # the module can be materialized again afterwards
    with materialized_params(module):
        pass
    assert calls[-2:] == [('allocate', 'a'), ('offload', 'a')]


def test_materialized_params_multiple_modules():
    calls = []
    modules = [_offloaded_linear(calls, n) for n in 'ab'] + [nn.Linear(4, 4)]
    with materialized_params(*modules):
        assert calls == [('allocate', 'a'), ('allocate', 'b')]
    assert calls[2:] == [('offload', 'b'), ('offload', 'a')]
