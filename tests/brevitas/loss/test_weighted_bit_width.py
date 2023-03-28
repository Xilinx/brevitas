# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.inject.defaults import Int8ActPerTensorFloat
from brevitas.loss import ActivationBitWidthWeightedBySize
from brevitas.loss import MEGA
from brevitas.loss import QuantLayerOutputBitWidthWeightedByOps
from brevitas.loss import WeightBitWidthWeightedBySize
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear


def test_weight_bit_width_weighted_by_size():
    model = QuantLinear(
        out_features=6,
        in_features=5,
        bias=False,
        weight_bit_width_impl_type='parameter',
        weight_bit_width=4)
    loss = WeightBitWidthWeightedBySize(model)
    out = model(torch.randn(2, 5, 5))
    assert loss.tot_num_elements == 30
    assert loss.retrieve() == 4.0


def test_act_bit_width_weighted_by_size():
    model = QuantIdentity(bit_width_impl_type='parameter', bit_width=3)
    loss = ActivationBitWidthWeightedBySize(model)
    out = model(torch.randn(2, 5, 5))
    assert loss.tot_num_elements == 25
    assert loss.retrieve() == 3.0


def test_output_bit_weighted_by_ops():
    model = QuantLinear(
        out_features=6,
        in_features=5,
        bias=False,
        input_quant=Int8ActPerTensorFloat,
        weight_bit_width_impl_type='parameter',
        return_quant_tensor=True)
    loss = QuantLayerOutputBitWidthWeightedByOps(model)
    out = model(torch.randn(2, 4, 5))
    assert loss.tot_num_elements == 24 * 10 / MEGA
    assert loss.retrieve() == out.bit_width
