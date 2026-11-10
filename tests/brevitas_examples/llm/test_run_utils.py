# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
import torch.nn as nn

from brevitas_examples.llm.llm_quant.run_utils import fix_rewriter


class _Rewriter:

    def __init__(self, module):
        self.old_module_instance = module


def _two_versions():
    # Two versions of the same model sharing the weight storage, as the eager and the
    # dynamo-exported models do
    eager = nn.Sequential(nn.Linear(16, 32), nn.Linear(32, 16), nn.Linear(64, 16))
    other = nn.Sequential(*[nn.Linear(1, 1) for _ in range(3)])
    for i in range(3):
        other[i].weight = nn.Parameter(eager[i].weight.data)
    return eager, other


def test_fix_rewriter_remaps_by_storage():
    eager, other = _two_versions()
    rewriters = fix_rewriter([_Rewriter(m) for m in other], eager, 'weight')
    assert [r.old_module_instance for r in rewriters] == list(eager)


def test_fix_rewriter_rejects_meta_tensors():
    # Offloaded weights are on the meta device, where data_ptr() is 0 for every tensor: matching
    # them by storage would silently remap every rewriter to the first offloaded module
    eager, other = _two_versions()
    for m in list(eager) + list(other):
        m.weight = nn.Parameter(torch.empty_like(m.weight, device='meta'))
    with pytest.raises(RuntimeError, match="meta device"):
        fix_rewriter([_Rewriter(other[2])], eager, 'weight')
