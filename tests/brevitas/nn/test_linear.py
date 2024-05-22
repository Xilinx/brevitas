# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from brevitas.nn import QuantLinear
from brevitas.quant import Int32Bias
from brevitas.quant_tensor import IntQuantTensor

OUTPUT_FEATURES = 10
INPUT_FEATURES = 5
BIT_WIDTH = 5


class TestQuantLinearInit:

    def test_module_init_defaults(self):
        mod = QuantLinear(out_features=OUTPUT_FEATURES, in_features=INPUT_FEATURES, bias=False)
        assert mod

    def test_module_init_bias_fp(self):
        mod = QuantLinear(out_features=OUTPUT_FEATURES, in_features=INPUT_FEATURES, bias=True)
        assert mod

    def test_module_init_bias_int(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True,
            bias_quant=Int32Bias)
        assert mod

    def test_module_init_scale_impl_type_override(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True,
            weight_scaling_impl_type='HE')
        assert mod.weight_quant.scale()


class TestQuantLinearFwd:

    def test_forward_defaults(self):
        mod = QuantLinear(out_features=OUTPUT_FEATURES, in_features=INPUT_FEATURES, bias=True)
        x = torch.rand(size=(3, INPUT_FEATURES))
        assert mod(x) is not None

    def test_forward_bias_fp(self):
        mod = QuantLinear(out_features=OUTPUT_FEATURES, in_features=INPUT_FEATURES, bias=True)
        x = torch.rand(size=(3, INPUT_FEATURES))
        assert mod(x) is not None

    def test_forward_bias_int(self):
        mod = QuantLinear(
            out_features=OUTPUT_FEATURES,
            in_features=INPUT_FEATURES,
            bias=True,
            bias_quant=Int32Bias)
        x = IntQuantTensor(
            torch.rand(size=(3, INPUT_FEATURES)),
            torch.tensor(1.0),
            torch.tensor(0.0),
            torch.tensor(3),
            signed=True,
            training=False)
        assert mod(x) is not None
