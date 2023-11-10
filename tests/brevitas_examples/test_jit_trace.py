# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import ExitStack

import pytest
import torch

from brevitas_examples.bnn_pynq.models import model_with_cfg

FC_INPUT_SIZE = (1, 1, 28, 28)
CNV_INPUT_SIZE = (1, 3, 32, 32)

MAX_WBITS = 2
MAX_ABITS = 2


@pytest.mark.parametrize("size", ["TFC", "SFC", "LFC"])
@pytest.mark.parametrize("wbits", [1, MAX_WBITS])
@pytest.mark.parametrize("abits", [1, MAX_ABITS])
def test_brevitas_fc_jit_trace(size, wbits, abits):
    if size == "LFC" and wbits == 2 and abits == 2:
        pytest.skip(f"No LFC_{MAX_WBITS}W{MAX_ABITS}A present.")
    if wbits > abits:
        pytest.skip("No wbits > abits cases.")
    nname = f"{size}_{wbits}W{abits}A"
    fc, _ = model_with_cfg(nname.lower(), pretrained=False)
    fc.train(False)
    input_tensor = torch.randn(FC_INPUT_SIZE)
    traced_model = torch.jit.trace(fc, input_tensor)
    out_traced = traced_model(input_tensor)
    out = fc(input_tensor)
    assert out.isclose(out_traced).all().item()


@pytest.mark.parametrize("wbits", [1, MAX_WBITS])
@pytest.mark.parametrize("abits", [1, MAX_ABITS])
def test_brevitas_cnv_jit_trace(wbits, abits):
    if wbits > abits:
        pytest.skip("No wbits > abits cases.")
    nname = f"CNV_{wbits}W{abits}A"
    cnv, _ = model_with_cfg(nname.lower(), pretrained=False)
    cnv.train(False)
    input_tensor = torch.randn(CNV_INPUT_SIZE)
    traced_model = torch.jit.trace(cnv, input_tensor)
    out_traced = traced_model(input_tensor)
    out = cnv(input_tensor)
    assert out.isclose(out_traced).all().item()
