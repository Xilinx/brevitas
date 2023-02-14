# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from inspect import getfullargspec

import torch

from brevitas.fx import value_trace
from brevitas.graph import EqualizeGraph

from .equalization_fixtures import *

SEED = 123456
IN_SIZE = (16,3,224,224)
ATOL = 1e-3


def test_models(toy_model):
    model = toy_model()
    inp = torch.randn(IN_SIZE)

    input_name = getfullargspec(model.forward)[0][0]
    model.eval()
    expected_out = model(inp)
    model = value_trace(model, {input_name: inp})
    model, regions = EqualizeGraph(3, return_regions=True).apply(model)

    out = model(inp)
    assert len(regions) > 0
    assert torch.allclose(expected_out, out, atol=ATOL)
