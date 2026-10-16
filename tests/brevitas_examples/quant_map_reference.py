# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Frozen reference tables for the quantizer-builder parity tests.

Historically ``generate_quantizers`` selected its quantizers by indexing the
static ``WEIGHT_QUANT_MAP`` / ``INPUT_QUANT_MAP`` tables. Those lookups were
replaced by :func:`create_weight_quantizer` / :func:`create_input_quantizer`
(see ``brevitas_examples.common.quantizer_builder``). The tables live on here as
frozen test data so :mod:`test_quant_map` can keep pinning the builder output to
the known-good reference classes, leaf by leaf.
"""

from brevitas.quant.fixed_point import Int8ActPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8ActPerTensorFixedPointMSE
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPointMSE
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPointMSE
from brevitas.quant.float import Fp8e4m3Act
from brevitas.quant.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.float import Fp8e4m3WeightPerChannelFloat
from brevitas.quant.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerChannelFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerTensorFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloat
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3WeightMSE
from brevitas.quant.mx_quant_ocp import MXInt8Act
from brevitas.quant.mx_quant_ocp import MXInt8Weight
from brevitas.quant.mx_quant_ocp import MXInt8WeightMSE
from brevitas.quant.mx_quant_ocp import ShiftedMXUInt8Weight
from brevitas.quant.mx_quant_ocp import ShiftedMXUInt8WeightMSE
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8ActPerTensorFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatHQO
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloatHQO
from brevitas.quant.scaled_int import Int8WeightPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8ActPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightGroupQuantFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerGroupFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3DynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import FP8e4m3FNUZDynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3FNUZDynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPDynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import FP8e4m3OCPDynamicActPerRowFixedPoint
from brevitas_examples.common.generative.quantizers import FP8e4m3OCPDynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPDynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPWeightPerChannelFixedPointMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPWeightPerChannelFloatMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPWeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import Fp8e4m3WeightPerChannelFloatMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3WeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerRowFixedPoint
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import Int8DynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import IntWeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import IntWeightSymmetricGroupQuantMSE
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerGroupFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerRowFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8DynamicActPerTensorFloat
from brevitas_examples.common.generative.quantizers import ShiftedUint8WeightGroupQuantFloatMSE

WEIGHT_QUANT_MAP = {
    'int': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloat, 'asym': ShiftedUint8WeightPerTensorFloat},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloat, 'asym': ShiftedUint8WeightPerChannelFloat},
                'per_group': {
                    'sym': IntWeightSymmetricGroupQuant,
                    'asym': ShiftedUint8WeightGroupQuantFloat}},
            'mse': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloatMSE,
                    'asym': ShiftedUint8WeightPerTensorFloatMSE},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloatMSE,
                    'asym': ShiftedUint8WeightPerChannelFloatMSE},
                'per_group': {
                    'sym': IntWeightSymmetricGroupQuantMSE,
                    'asym': ShiftedUint8WeightGroupQuantFloatMSE}},
            'hqo': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFloatHQO,
                    'asym': ShiftedUint8WeightPerTensorFloatHQO},
                'per_channel': {
                    'sym': Int8WeightPerChannelFloatHQO,
                    'asym': ShiftedUint8WeightPerChannelFloatHQO},
                'per_group': {
                    'asym': ShiftedUint8WeightPerGroupFloatHQO}},},
        'po2_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFixedPoint},
                'per_channel': {
                    'sym': Int8WeightPerChannelFixedPoint},
                'per_group': {
                    'sym': MXInt8Weight, 'asym': ShiftedMXUInt8Weight}},
            'mse': {
                'per_tensor': {
                    'sym': Int8WeightPerTensorFixedPointMSE},
                'per_channel': {
                    'sym': Int8WeightPerChannelFixedPointMSE},
                'per_group': {
                    'sym': MXInt8WeightMSE, 'asym': ShiftedMXUInt8WeightMSE}}}},
    'float': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3WeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3WeightPerChannelFloat},
                'per_group': {
                    'sym': Fp8e4m3WeightSymmetricGroupQuant}},
            'mse': {
                'per_channel': {
                    'sym': Fp8e4m3WeightPerChannelFloatMSE}}}},
    'float_ocp': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3OCPWeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3OCPWeightPerChannelFloat},
                'per_group': {
                    'sym': Fp8e4m3OCPWeightSymmetricGroupQuant}},
            'mse': {
                'per_channel': {
                    'sym': Fp8e4m3OCPWeightPerChannelFloatMSE}}},
        'po2_scale': {
            'stats': {
                'per_group': {
                    'sym': MXFloat8e4m3Weight}},
            'mse': {
                'per_channel': {
                    'sym': Fp8e4m3OCPWeightPerChannelFixedPointMSE},
                'per_group': {
                    'sym': MXFloat8e4m3WeightMSE}}}},
    'float_fnuz': {
        'float_scale': {
            'stats': {
                'per_tensor': {
                    'sym': Fp8e4m3FNUZWeightPerTensorFloat},
                'per_channel': {
                    'sym': Fp8e4m3FNUZWeightPerChannelFloat}}}}}

INPUT_QUANT_MAP = {
    'int': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFloat, 'asym': ShiftedUint8ActPerTensorFloat}},
                'mse': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFloatMSE,
                        'asym': ShiftedUint8ActPerTensorFloatMSE}}},
            'po2_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFixedPoint}},
                'mse': {
                    'per_tensor': {
                        'sym': Int8ActPerTensorFixedPointMSE}}}},
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Int8DynamicActPerTensorFloat,
                        'asym': ShiftedUint8DynamicActPerTensorFloat},
                    'per_row': {
                        'sym': Int8DynamicActPerRowFloat,
                        'asym': ShiftedUint8DynamicActPerRowFloat},
                    'per_group': {
                        'sym': Int8DynamicActPerGroupFloat,
                        'asym': ShiftedUint8DynamicActPerGroupFloat}}},
            'po2_scale': {
                'stats': {
                    'per_row': {
                        'sym': Int8DynamicActPerRowFixedPoint,},
                    'per_group': {
                        'sym': MXInt8Act}}}}},
    'float': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3ActPerTensorFloat}}}},
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_group': {
                        'sym': Fp8e4m3DynamicActPerGroupFloat}}}},
        'no_scale': {
            'sym': Fp8e4m3Act,}},
    'float_ocp': {
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3OCPActPerTensorFloat}}}},
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3OCPDynamicActPerTensorFloat},
                    'per_row': {
                        'sym': FP8e4m3OCPDynamicActPerRowFloat},
                    'per_group': {
                        'sym': Fp8e4m3OCPDynamicActPerGroupFloat}}},
            'po2_scale': {
                'stats': {
                    'per_row': {
                        'sym': FP8e4m3OCPDynamicActPerRowFixedPoint},
                    'per_group': {
                        'sym': MXFloat8e4m3Act}}}}},
    'float_fnuz': {
        'dynamic': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3FNUZDynamicActPerTensorFloat},
                    'per_row': {
                        'sym': FP8e4m3FNUZDynamicActPerRowFloat}}}},
        'static': {
            'float_scale': {
                'stats': {
                    'per_tensor': {
                        'sym': Fp8e4m3FNUZActPerTensorFloat}}}}}}
