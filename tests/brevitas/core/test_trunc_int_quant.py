# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import pytest_cases
import torch

from brevitas.core.bit_width import BitWidthConst
from brevitas.core.function_wrapper import Identity
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.function_wrapper import TensorClamp
from brevitas.core.quant import TruncIntQuant
from brevitas.core.restrict_val import PowerOfTwoRestrictValue
from brevitas.core.scaling import PowerOfTwoIntScaling
from brevitas.core.scaling import RuntimeStatsScaling
from brevitas.core.scaling import TruncMsbScaling
from brevitas.core.scaling import TruncScalingWrapper
from brevitas.core.stats import AbsMax
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
            "defaults_int+_overflow",
            "defaults_int-_max",
            "defaults_uint_underflow",
            "defaults_int_underflow",
            "defaults_uint_ulp",
            "defaults_int_ulp",
            "abxmax_uint_overflow",
            "abxmax_int+_overflow",
            "abxmax_int-_overflow",
            "abxmax_uint_underflow",
            "abxmax_int_underflow",
            "abxmax_uint_ulp",
            "abxmax_int_ulp",
        ],
        params=[
            { # defaults_uint_overflow
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
            }, { # defaults_int+_overflow
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
            }, { # defaults_int-_max
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                },
                "train_args": {
                    "x": torch.tensor([-128.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "eval_args": {
                    "x": torch.tensor([-128.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([-128.]),
                    "scale": torch.tensor([16.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # defaults_uint_underflow
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                },
                "train_args": {
                    "x": torch.tensor([8.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "eval_args": {
                    "x": torch.tensor([8.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([0.]),
                    "scale": torch.tensor([16.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # defaults_int_underflow
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                },
                "train_args": {
                    "x": torch.tensor([-8.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "eval_args": {
                    "x": torch.tensor([-8.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([0.]),
                    "scale": torch.tensor([16.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # defaults_uint_ulp
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                },
                "train_args": {
                    "x": torch.tensor([9.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "eval_args": {
                    "x": torch.tensor([9.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([16.]),
                    "scale": torch.tensor([16.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # defaults_int_ulp
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                },
                "train_args": {
                    "x": torch.tensor([-9.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "eval_args": {
                    "x": torch.tensor([-9.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([-16.]),
                    "scale": torch.tensor([16.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # abxmax_uint_overflow
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                    "trunc_scaling_impl": TruncScalingWrapper(
                        trunc_int_scaling_impl=PowerOfTwoIntScaling(),
                        scaling_impl=RuntimeStatsScaling(
                            scaling_stats_impl=AbsMax(),
                            scaling_stats_input_view_shape_impl=Identity(),
                            scaling_shape=(1,),
                            scaling_stats_momentum=1.0,
                            restrict_scaling_impl=PowerOfTwoRestrictValue(),
                        )
                    ),
                },
                "train_args": {
                    "x": torch.tensor([128.]),
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
                    "y": torch.tensor([120.]),
                    "scale": torch.tensor([8.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # abxmax_int+_overflow
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                    "trunc_scaling_impl": TruncScalingWrapper(
                        trunc_int_scaling_impl=PowerOfTwoIntScaling(),
                        scaling_impl=RuntimeStatsScaling(
                            scaling_stats_impl=AbsMax(),
                            scaling_stats_input_view_shape_impl=Identity(),
                            scaling_shape=(1,),
                            scaling_stats_momentum=1.0,
                            restrict_scaling_impl=PowerOfTwoRestrictValue(),
                        )
                    ),
                },
                "train_args": {
                    "x": torch.tensor([32.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "eval_args": {
                    "x": torch.tensor([64.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([28.]),
                    "scale": torch.tensor([4.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # abxmax_int-_overflow
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                    "trunc_scaling_impl": TruncScalingWrapper(
                        trunc_int_scaling_impl=PowerOfTwoIntScaling(),
                        scaling_impl=RuntimeStatsScaling(
                            scaling_stats_impl=AbsMax(),
                            scaling_stats_input_view_shape_impl=Identity(),
                            scaling_shape=(1,),
                            scaling_stats_momentum=1.0,
                            restrict_scaling_impl=PowerOfTwoRestrictValue(),
                        )
                    ),
                },
                "train_args": {
                    "x": torch.tensor([-16.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "eval_args": {
                    "x": torch.tensor([-32.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([-16.]),
                    "scale": torch.tensor([2.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # abxmax_uint_underflow
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                    "trunc_scaling_impl": TruncScalingWrapper(
                        trunc_int_scaling_impl=PowerOfTwoIntScaling(),
                        scaling_impl=RuntimeStatsScaling(
                            scaling_stats_impl=AbsMax(),
                            scaling_stats_input_view_shape_impl=Identity(),
                            scaling_shape=(1,),
                            scaling_stats_momentum=1.0,
                            restrict_scaling_impl=PowerOfTwoRestrictValue(),
                        )
                    ),
                },
                "train_args": {
                    "x": torch.tensor([15.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "eval_args": {
                    "x": torch.tensor([.5]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([0.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # abxmax_int_underflow
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                    "trunc_scaling_impl": TruncScalingWrapper(
                        trunc_int_scaling_impl=PowerOfTwoIntScaling(),
                        scaling_impl=RuntimeStatsScaling(
                            scaling_stats_impl=AbsMax(),
                            scaling_stats_input_view_shape_impl=Identity(),
                            scaling_shape=(1,),
                            scaling_stats_momentum=1.0,
                            restrict_scaling_impl=PowerOfTwoRestrictValue(),
                        )
                    ),
                },
                "train_args": {
                    "x": torch.tensor([-8.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "eval_args": {
                    "x": torch.tensor([-.5]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([0.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # abxmax_uint_ulp
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                    "trunc_scaling_impl": TruncScalingWrapper(
                        trunc_int_scaling_impl=PowerOfTwoIntScaling(),
                        scaling_impl=RuntimeStatsScaling(
                            scaling_stats_impl=AbsMax(),
                            scaling_stats_input_view_shape_impl=Identity(),
                            scaling_shape=(1,),
                            scaling_stats_momentum=1.0,
                            restrict_scaling_impl=PowerOfTwoRestrictValue(),
                        )
                    ),
                },
                "train_args": {
                    "x": torch.tensor([31.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "eval_args": {
                    "x": torch.tensor([2.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": False,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([2.]),
                    "scale": torch.tensor([2.]),
                    "zero_point": torch.tensor([0.]),
                    "bit_width": torch.tensor([4.]),
                },
            }, { # abxmax_int_ulp
                "init_args": {
                    "bit_width_impl": BitWidthConst(4),
                    "float_to_int_impl": RoundSte(),
                    "trunc_scaling_impl": TruncScalingWrapper(
                        trunc_int_scaling_impl=PowerOfTwoIntScaling(),
                        scaling_impl=RuntimeStatsScaling(
                            scaling_stats_impl=AbsMax(),
                            scaling_stats_input_view_shape_impl=Identity(),
                            scaling_shape=(1,),
                            scaling_stats_momentum=1.0,
                            restrict_scaling_impl=PowerOfTwoRestrictValue(),
                        )
                    ),
                },
                "train_args": {
                    "x": torch.tensor([-64.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "eval_args": {
                    "x": torch.tensor([-5.]),
                    "scale": torch.tensor([1.]),
                    "zero_point": torch.tensor([0.]),
                    "input_bit_width": torch.tensor([8.]),
                    "signed": True,
                },
                "result": { # Result needs to match the order of the output tuple
                    "y": torch.tensor([-8.]),
                    "scale": torch.tensor([8.]),
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
                assert torch.allclose(expected_result[k], y[i], rtol=0.0, atol=0.0, equal_nan=False), f"Expected result[{k}]: {expected_result[k]}, result: {y[i]}"
