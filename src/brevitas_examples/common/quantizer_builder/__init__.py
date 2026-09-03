"""
Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

Programmatic quantizer builder.

Assembles brevitas quantizer injectors from orthogonal quantization axes
(format, scale precision / type, param method, granularity, ...) via a Builder +
Component design, reproducing the entries of ``WEIGHT_QUANT_MAP`` /
``INPUT_QUANT_MAP`` without the static lookup tables.

The public entry point is the ``build_quantizer(builder_cls, ...)`` factory (with
``builder_cls`` being :class:`WeightQuantizerBuilder` / :class:`InputQuantizerBuilder`);
the remaining exports (config, enums, base classes) support customisation via
``extra_components``.
"""
from brevitas_examples.common.quantizer_builder.builder import build_quantizer
from brevitas_examples.common.quantizer_builder.builder import QuantizerBuilder
from brevitas_examples.common.quantizer_builder.components import QuantScaleRestrictComponent
from brevitas_examples.common.quantizer_builder.core import Component
from brevitas_examples.common.quantizer_builder.core import config_from_flat_args
from brevitas_examples.common.quantizer_builder.core import Contribution
from brevitas_examples.common.quantizer_builder.core import FloatFormatConfig
from brevitas_examples.common.quantizer_builder.core import FormatConfig
from brevitas_examples.common.quantizer_builder.core import IntFormatConfig
from brevitas_examples.common.quantizer_builder.core import QuantizerConfig
from brevitas_examples.common.quantizer_builder.core import QuantScaleQuantizerConfig
from brevitas_examples.common.quantizer_builder.input import InputQuantizerBuilder
from brevitas_examples.common.quantizer_builder.mixins import FloatFormat
from brevitas_examples.common.quantizer_builder.mixins import ParamMethod
from brevitas_examples.common.quantizer_builder.mixins import QuantParamType
from brevitas_examples.common.quantizer_builder.mixins import ZeroPointImplType
from brevitas_examples.common.quantizer_builder.weight import default_scale_quantizer_config
from brevitas_examples.common.quantizer_builder.weight import QuantScaleWeightQuantizerBuilder
from brevitas_examples.common.quantizer_builder.weight import WeightQuantizerBuilder

__all__ = [
    "QuantizerBuilder",
    "WeightQuantizerBuilder",
    "QuantScaleWeightQuantizerBuilder",
    "QuantScaleRestrictComponent",
    "default_scale_quantizer_config",
    "InputQuantizerBuilder",
    "build_quantizer",
    "config_from_flat_args",
    "QuantizerConfig",
    "QuantScaleQuantizerConfig",
    "IntFormatConfig",
    "FloatFormatConfig",
    "FormatConfig",
    "Component",
    "Contribution",
    "FloatFormat",
    "ParamMethod",
    "QuantParamType",
    "ZeroPointImplType",]
