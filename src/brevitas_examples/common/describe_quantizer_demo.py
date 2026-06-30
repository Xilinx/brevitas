"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Standalone demo: build a few weight/input quantizers (same args/kwargs as the
specs in tests/brevitas_examples/test_quantizer_builder.py and
test_input_quantizer_builder.py) and print, for each, the injector attributes,
dependency kinds, and -- for ``@value`` functions -- the args they require and
what they resolve to.

Run with: python src/brevitas_examples/common/describe_quantizer_demo.py
"""
from brevitas.core.stats.stats_op import AbsMax
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas_examples.common.quantizer_builder import build_input_quantizer
from brevitas_examples.common.quantizer_builder import build_weight_quantizer
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder import ScaleType

BIT_WIDTH = 8
GROUP_SIZE = 8
SCALING_MIN_VAL = 1e-10

# Same builder args/kwargs as the test specs. Each value pairs the builder
# factory with the kwargs to pass it.
WEIGHT_SPECS = {
    "int_per_channel_sym": {
        "quant_type": QuantType.INT,
        "quant_param_type": QuantParamType.SYM,
        "bit_width": BIT_WIDTH,
        "scaling_impl_type": ScalingImplType.STATS,
        "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
        "restrict_scaling_type": RestrictValueType.FP,
        "scaling_min_val": SCALING_MIN_VAL,
        "kwargs": {},},
    "int_per_group_asym": {
        "quant_type": QuantType.INT,
        "quant_param_type": QuantParamType.ASYM,
        "bit_width": BIT_WIDTH,
        "scaling_impl_type": ScalingImplType.STATS,
        "scaling_per_output_type": ScalingPerOutputType.GROUP,
        "restrict_scaling_type": RestrictValueType.FP,
        "scaling_min_val": SCALING_MIN_VAL,
        "kwargs": {
            "group_size": GROUP_SIZE},},
    "int_per_tensor_sym_mse": {
        "quant_type": QuantType.INT,
        "quant_param_type": QuantParamType.SYM,
        "bit_width": BIT_WIDTH,
        "scaling_param_method": ParamMethod.MSE,
        "scaling_per_output_type": ScalingPerOutputType.TENSOR,
        "restrict_scaling_type": RestrictValueType.FP,
        "scaling_min_val": SCALING_MIN_VAL,
        "kwargs": {},},
    "float_per_tensor_sym": {
        "quant_type": QuantType.FP,
        "quant_param_type": QuantParamType.SYM,
        "float_format": FloatFormat.FLOAT,
        "float_quant_format": "e4m3",
        "scaling_per_output_type": ScalingPerOutputType.TENSOR,
        "restrict_scaling_type": RestrictValueType.FP,
        "scaling_min_val": SCALING_MIN_VAL,
        "kwargs": {},},}

INPUT_SPECS = {
    "int_static_per_tensor_sym": {
        "quant_type": QuantType.INT,
        "quant_param_type": QuantParamType.SYM,
        "scale_type": ScaleType.STATIC,
        "bit_width": BIT_WIDTH,
        "scaling_per_output_type": ScalingPerOutputType.TENSOR,
        "restrict_scaling_type": RestrictValueType.FP,
        "scaling_min_val": SCALING_MIN_VAL,
        "kwargs": {},},
    "int_static_per_tensor_sym_mse": {
        "quant_type": QuantType.INT,
        "quant_param_type": QuantParamType.SYM,
        "scale_type": ScaleType.STATIC,
        "bit_width": BIT_WIDTH,
        "scaling_param_method": ParamMethod.MSE,
        "scaling_per_output_type": ScalingPerOutputType.TENSOR,
        "restrict_scaling_type": RestrictValueType.FP,
        "scaling_min_val": SCALING_MIN_VAL,
        "kwargs": {
            "scaling_mse_init_op": AbsMax},},}


def _header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    for name, builder_args in WEIGHT_SPECS.items():
        _header(f"[weight] {name}")
        build_weight_quantizer(**builder_args).describe_quantizer()

    for name, builder_args in INPUT_SPECS.items():
        _header(f"[input] {name}")
        build_input_quantizer(**builder_args).describe_quantizer()


if __name__ == "__main__":
    main()
