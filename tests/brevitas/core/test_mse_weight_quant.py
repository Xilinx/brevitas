# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas.nn.quant_linear import QuantLinear
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatMSE
from tests.conftest import SEED
from tests.marker import jit_disabled_for_local_loss

IN_FEATURES = 32
OUT_FEATURES = 16
ATOL = 1e-8

WEIGHT_QUANT_PAIRS = {
    'int8_per_channel': (Int8WeightPerChannelFloat, Int8WeightPerChannelFloatMSE),
    'shifted_uint8_per_channel':
        (ShiftedUint8WeightPerChannelFloat, ShiftedUint8WeightPerChannelFloatMSE),}


def weight_quant_mse(standard_quant, mse_quant):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(SEED)
    weight = torch.nn.Parameter(
        torch.randn((OUT_FEATURES, IN_FEATURES), generator=generator))
    inp = torch.randn((1, IN_FEATURES), generator=generator)

    standard_layer = QuantLinear(
        IN_FEATURES, OUT_FEATURES, bias=False, weight_quant=standard_quant)
    mse_layer = QuantLinear(IN_FEATURES, OUT_FEATURES, bias=False, weight_quant=mse_quant)
    # Both layers share the same float weight tensor
    standard_layer.weight = weight
    mse_layer.weight = weight

    # Forward pass initializes the (MSE) scales and zero-points
    standard_layer(inp)
    mse_layer(inp)

    standard_mse = ((weight - standard_layer.quant_weight().value) ** 2).mean()
    mse_mse = ((weight - mse_layer.quant_weight().value) ** 2).mean()
    return standard_mse, mse_mse


@pytest.mark.parametrize('quant_pair', WEIGHT_QUANT_PAIRS.values(), ids=WEIGHT_QUANT_PAIRS.keys())
@jit_disabled_for_local_loss()
def test_weight_quant_mse_le_standard(quant_pair):
    standard_quant, mse_quant = quant_pair
    standard_mse, mse_mse = weight_quant_mse(standard_quant, mse_quant)
    assert mse_mse <= standard_mse + ATOL, (
        f'MSE quantizer error {mse_mse:.3e} is larger than '
        f'standard quantizer error {standard_mse:.3e}')
