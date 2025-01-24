# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import pytest_cases
import torch

from brevitas.core.bit_width import BitWidthConst
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant import TruncIntQuant
from brevitas.core.scaling import TruncMsbScaling
from tests.brevitas.core.bit_width_fixture import *  # noqa
from tests.brevitas.core.int_quant_fixture import *  # noqa


def allexact(x, y):
    return np.allclose(x, y, rtol=0.0, atol=0.0, equal_nan=False)


class TestTruncIntQuantUnit:

    def test_trunc_int_quant_defaults(self, bit_width_const):
        trunc_int_quant = TruncIntQuant(
            bit_width_impl=bit_width_const, float_to_int_impl=RoundSte())
        assert isinstance(trunc_int_quant.tensor_clamp_impl, TensorClamp)
        assert isinstance(trunc_int_quant.trunc_scaling_impl, TruncMsbScaling)
        assert trunc_int_quant.narrow_range == False

    # yapf: disable
    @pytest_cases.fixture(
        ids=[
            "defaults_uint_overflow",
            "defaults_int_overflow",
        ],
        params=[
            {
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                },
                "train_args": {
                    "x": torch.tensor([255.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "eval_args": {
                    "x": torch.tensor([255.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([240.]),
                    "scale": torch.tensor([16.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, {
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                },
                "train_args": {
                    "x": torch.tensor([127.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "eval_args": {
                    "x": torch.tensor([127.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([112.]),
                    "scale": torch.tensor([16.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            },
        ],)
    # yapf: enable
    def trunc_int_quant_io_fixture(self, request):
        yield request.param

    def test_trunc_int_quant_io(self, caplog, trunc_int_quant_io_fixture):
        caplog.set_level(logging.INFO)
        test_cfg = trunc_int_quant_io_fixture
        init_args = test_cfg["init_args"]
        train_args = test_cfg["train_args"]
        eval_args = test_cfg["eval_args"]
        expected_result = test_cfg["result"]
        trunc_int_quant = TruncIntQuant(**init_args)
        trunc_int_quant.train()
        y = trunc_int_quant(**train_args)
        trunc_int_quant.eval()
        with torch.no_grad():
            y = trunc_int_quant(**eval_args)
            for i, k in enumerate(expected_result.keys()):
                assert torch.allclose(expected_result[k], y[i], rtol=0.0, atol=0.0, equal_nan=False), "Expected result[{k}]: {expected_result[k]}, result: {y[i]}"
