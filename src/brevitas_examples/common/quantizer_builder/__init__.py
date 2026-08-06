"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Programmatic quantizer builder.

Assembles brevitas quantizer injectors from orthogonal quantization axes
(format, scale precision / type, param method, granularity, ...) via a Builder +
Component design, reproducing the entries of ``WEIGHT_QUANT_MAP`` /
``INPUT_QUANT_MAP`` without the static lookup tables.

Public entry points are the ``build_weight_quantizer`` / ``build_input_quantizer``
factory shims; the remaining exports (config, enums, base classes) support
customisation via ``extra_components``.
"""
from brevitas_examples.common.quantizer_builder.builder import QuantizerBuilder
from brevitas_examples.common.quantizer_builder.core import Component
from brevitas_examples.common.quantizer_builder.core import config_from_flat_args
from brevitas_examples.common.quantizer_builder.core import Contribution
from brevitas_examples.common.quantizer_builder.core import FloatFormatConfig
from brevitas_examples.common.quantizer_builder.core import FormatConfig
from brevitas_examples.common.quantizer_builder.core import IntFormatConfig
from brevitas_examples.common.quantizer_builder.core import QuantizerConfig
from brevitas_examples.common.quantizer_builder.input import build_input_quantizer
from brevitas_examples.common.quantizer_builder.input import InputQuantizerBuilder
from brevitas_examples.common.quantizer_builder.mixins import FloatFormat
from brevitas_examples.common.quantizer_builder.mixins import ParamMethod
from brevitas_examples.common.quantizer_builder.mixins import QuantParamType
from brevitas_examples.common.quantizer_builder.mixins import ScaleType
from brevitas_examples.common.quantizer_builder.mixins import ZeroPointImplType
from brevitas_examples.common.quantizer_builder.weight import build_weight_quantizer
from brevitas_examples.common.quantizer_builder.weight import WeightQuantizerBuilder

__all__ = [
    "QuantizerBuilder",
    "WeightQuantizerBuilder",
    "InputQuantizerBuilder",
    "build_weight_quantizer",
    "build_input_quantizer",
    "config_from_flat_args",
    "QuantizerConfig",
    "IntFormatConfig",
    "FloatFormatConfig",
    "FormatConfig",
    "Component",
    "Contribution",
    "FloatFormat",
    "ParamMethod",
    "QuantParamType",
    "ScaleType",
    "ZeroPointImplType",]
