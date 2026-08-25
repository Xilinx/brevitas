# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from brevitas import config
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.nn import QuantLinear
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerChannelFixedPointMSE
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPoint
from brevitas.quant.fixed_point import Int8WeightPerTensorFixedPointMSE
from brevitas.quant.float import Fp8e4m3WeightPerChannelFloat
from brevitas.quant.float import Fp8e4m3WeightPerChannelFloatMSE
from brevitas.quant.float import Fp8e4m3WeightPerTensorFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerChannelFloat
from brevitas.quant.float_quant_fnuz import Fp8e4m3FNUZWeightPerTensorFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloat
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerChannelFloatMSE
from brevitas.quant.float_quant_ocp import Fp8e4m3OCPWeightPerTensorFloat
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3Weight
from brevitas.quant.mx_quant_ocp import MXFloat8e4m3WeightMSE
from brevitas.quant.mx_quant_ocp import MXInt8Weight
from brevitas.quant.mx_quant_ocp import MXInt8WeightMSE
from brevitas.quant.mx_quant_ocp import ShiftedMXUInt8Weight
from brevitas.quant.mx_quant_ocp import ShiftedMXUInt8WeightMSE
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatHQO
from brevitas.quant.scaled_int import Int8WeightPerChannelFloatMSE
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloatHQO
from brevitas.quant.scaled_int import Int8WeightPerTensorFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightGroupQuantFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloatMSE
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerGroupFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatHQO
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloatMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPWeightPerChannelFixedPointMSE
from brevitas_examples.common.generative.quantizers import Fp8e4m3OCPWeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import Fp8e4m3WeightSymmetricGroupQuant
from brevitas_examples.common.generative.quantizers import IntWeightSymmetricGroupQuant
from brevitas_examples.common.quantizer_builder import build_weight_quantizer
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType

# Keep the model small and deterministic so that weight-quant outputs are
# directly comparable between the reference quantizer and the builder.
torch.manual_seed(0)

IN_FEATURES = 32
OUT_FEATURES = 16
GROUP_SIZE = 8
BIT_WIDTH = 8
# Matches the ``scaling_min_val`` carried by the reference weight quantizers
# through MaxStatsScaling / MinMaxStatsScaling (brevitas.quant.base).
SCALING_MIN_VAL = 1e-10

# A builder spec describes how to reproduce a given reference quantizer class
# from WEIGHT_QUANT_MAP['int']['float_scale']['stats'] through the generic
# QuantizerBuilder. Each spec carries:
#   - ``ref``: the reference quantizer class (the WEIGHT_QUANT_MAP leaf)
#   - ``builder_args``: kwargs passed to QuantizerBuilder.__init__
#   - ``layer_kwargs``: extra kwargs for the QuantLinear wrapping the quantizer
#     (e.g. group_size for groupwise quantization)
#
# The ``kwargs`` entry of ``builder_args`` carries the directives that are not
# first-class builder parameters but are still required to match the reference
# quantizer (e.g. narrow_range / quantize_zero_point overrides for MX).
#
# Signedness, narrow range, the scaling stats op and zero-point handling are
# now driven by the builder's sym/asym mixins, selected via ``quant_param_type``
# (QuantParamType.SYM / QuantParamType.ASYM). The mixins set scaling_stats_op
# (MAX for sym, MIN_MAX for asym) and wire the zero-point implementation, so
# those no longer need to be passed explicitly. The proxy class is also derived
# automatically by the builder from ``scaling_per_output_type``.
BUILDER_SPECS = {
    "int_per_tensor_sym": {
        "ref": Int8WeightPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_per_tensor_asym": {
        "ref": ShiftedUint8WeightPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_per_channel_sym": {
        "ref": Int8WeightPerChannelFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_per_channel_asym": {
        "ref": ShiftedUint8WeightPerChannelFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_per_group_sym": {
        "ref": IntWeightSymmetricGroupQuant,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    "int_per_group_asym": {
        "ref": ShiftedUint8WeightGroupQuantFloat,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            "scaling_impl_type": ScalingImplType.STATS,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    # ----------------------------------------------------------------------
    # MSE param method: WEIGHT_QUANT_MAP['int']['float_scale']['mse'].
    # The scale is learned/initialized from an MSE local loss, selected on the
    # builder side via scaling_param_method=ParamMethod.MSE. For the asymmetric
    # quantizers the zero-point is *also* learned from an MSE local loss,
    # selected via zero_point_param_method=ParamMethod.MSE.
    # ----------------------------------------------------------------------
    "int_per_tensor_sym_mse": {
        "ref": Int8WeightPerTensorFloatMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_per_channel_sym_mse": {
        "ref": Int8WeightPerChannelFloatMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_per_tensor_asym_mse": {
        "ref": ShiftedUint8WeightPerTensorFloatMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "zero_point_param_method": ParamMethod.MSE,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_per_channel_asym_mse": {
        "ref": ShiftedUint8WeightPerChannelFloatMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "zero_point_param_method": ParamMethod.MSE,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    # ----------------------------------------------------------------------
    # HQO param method: WEIGHT_QUANT_MAP['int']['float_scale']['hqo'].
    #
    # For *symmetric* HQO the scale is learned/initialized from a Half-Quadratic
    # Optimization local loss (scaling_param_method=ParamMethod.HQO).
    #
    # For *asymmetric* HQO only the zero-point is learned via HQO
    # (zero_point_param_method=ParamMethod.HQO); the scale stays a regular
    # MinMax stats scale (scaling_param_method defaults to STATS). The reference
    # quantizers also set quantize_zero_point=False, which we mirror via kwargs.
    # ----------------------------------------------------------------------
    "int_per_tensor_sym_hqo": {
        "ref": Int8WeightPerTensorFloatHQO,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.HQO,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_per_channel_sym_hqo": {
        "ref": Int8WeightPerChannelFloatHQO,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.HQO,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_per_tensor_asym_hqo": {
        "ref": ShiftedUint8WeightPerTensorFloatHQO,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            "zero_point_param_method": ParamMethod.HQO,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "quantize_zero_point": False,},},
        "layer_kwargs": {},},
    "int_per_channel_asym_hqo": {
        "ref": ShiftedUint8WeightPerChannelFloatHQO,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            "zero_point_param_method": ParamMethod.HQO,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "quantize_zero_point": False,},},
        "layer_kwargs": {},},
    "int_per_group_asym_hqo": {
        "ref": ShiftedUint8WeightPerGroupFloatHQO,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            "zero_point_param_method": ParamMethod.HQO,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "quantize_zero_point": False,
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    # ----------------------------------------------------------------------
    # po2_scale: WEIGHT_QUANT_MAP['int']['po2_scale']. The scale is restricted
    # to a power of two, selected on the builder side via
    # restrict_scaling_type=RestrictValueType.POWER_OF_TWO. The per_tensor /
    # per_channel variants are plain fixed-point quantizers; the per_group
    # variants are MX quantizers.
    # ----------------------------------------------------------------------
    "int_po2_per_tensor_sym": {
        "ref": Int8WeightPerTensorFixedPoint,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_po2_per_channel_sym": {
        "ref": Int8WeightPerChannelFixedPoint,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_po2_per_tensor_sym_mse": {
        "ref": Int8WeightPerTensorFixedPointMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "int_po2_per_channel_sym_mse": {
        "ref": Int8WeightPerChannelFixedPointMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},  # MX (groupwise po2) quantizers:
    # WEIGHT_QUANT_MAP['int']['po2_scale'][...]['per_group'].
    "int_po2_per_group_sym": {
        "ref": MXInt8Weight,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                # MX int uses IntQuant (narrow_range=False), unlike the
                # NarrowIntQuant-based per_tensor/per_channel sym quantizers.
                "narrow_range": False,
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    "int_po2_per_group_asym": {
        "ref": ShiftedMXUInt8Weight,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    "int_po2_per_group_sym_mse": {
        "ref": MXInt8WeightMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.SYM,
            "bit_width": BIT_WIDTH,
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                # MX int uses IntQuant (narrow_range=False).
                "narrow_range": False,
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    "int_po2_per_group_asym_mse": {
        "ref": ShiftedMXUInt8WeightMSE,
        "builder_args": {
            "quant_type": QuantType.INT,
            "quant_param_type": QuantParamType.ASYM,
            "bit_width": BIT_WIDTH,
            # MSEAsymmetricScale only makes the *scale* MSE-based; the zero-point
            # stays a plain MinMax stats zero-point (from ShiftedMinUintQuant).
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},
        # The reference brevitas quantizer ShiftedMXUInt8WeightMSE is itself
        # broken for groupwise: MSEAsymmetricScale sets
        # scaling_stats_input_view_shape_impl=Identity, which (via
        # ShiftedMinUintQuant) also becomes the zero-point stats view, while the
        # zero-point stats still reduce over the group dim (stats_reduce_dim=2).
        # This raises an IndexError inside the *reference* forward, so there is
        # nothing for the builder to match against.
        "xfail":
            "Reference ShiftedMXUInt8WeightMSE crashes for groupwise "
            "(zero-point stats view is Identity but reduces over group dim).",},
    # ----------------------------------------------------------------------
    # float / float_scale: WEIGHT_QUANT_MAP['float']['float_scale'].
    # FP8 e4m3 weight quantizers (signed, symmetric). Selected on the builder
    # side via quant_type=QuantType.FP, float_format=FloatFormat.FLOAT and
    # float_quant_format='e4m3'.
    # ----------------------------------------------------------------------
    "float_per_tensor_sym": {
        "ref": Fp8e4m3WeightPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.FLOAT,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "float_per_channel_sym": {
        "ref": Fp8e4m3WeightPerChannelFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.FLOAT,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "float_per_group_sym": {
        "ref": Fp8e4m3WeightSymmetricGroupQuant,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.FLOAT,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    "float_per_channel_sym_mse": {
        "ref": Fp8e4m3WeightPerChannelFloatMSE,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.FLOAT,
            "float_quant_format": "e4m3",
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    # ----------------------------------------------------------------------
    # float_ocp / float_scale: WEIGHT_QUANT_MAP['float_ocp']['float_scale'].
    # ----------------------------------------------------------------------
    "float_ocp_per_tensor_sym": {
        "ref": Fp8e4m3OCPWeightPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "float_ocp_per_channel_sym": {
        "ref": Fp8e4m3OCPWeightPerChannelFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "float_ocp_per_group_sym": {
        "ref": Fp8e4m3OCPWeightSymmetricGroupQuant,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    "float_ocp_per_channel_sym_mse": {
        "ref": Fp8e4m3OCPWeightPerChannelFloatMSE,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    # ----------------------------------------------------------------------
    # float_ocp / po2_scale (MX float):
    # WEIGHT_QUANT_MAP['float_ocp']['po2_scale']. MX float is OCP-only and uses
    # a power-of-two group scale; on the builder side selected via
    # quant_type=FP, float_format=OCP and restrict_scaling_type=POWER_OF_TWO.
    # ----------------------------------------------------------------------
    "float_ocp_po2_per_group_sym": {
        "ref": MXFloat8e4m3Weight,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    "float_ocp_po2_per_group_sym_mse": {
        "ref": MXFloat8e4m3WeightMSE,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.GROUP,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {
                "group_size": GROUP_SIZE,},},
        "layer_kwargs": {
            "weight_group_size": GROUP_SIZE},},
    "float_ocp_po2_per_channel_sym_mse": {
        "ref": Fp8e4m3OCPWeightPerChannelFixedPointMSE,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.OCP,
            "float_quant_format": "e4m3",
            "scaling_param_method": ParamMethod.MSE,
            "scaling_impl_type": ScalingImplType.PARAMETER_FROM_STATS,
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.POWER_OF_TWO,
            "scaling_min_val": SCALING_MIN_VAL,
            # Per-channel power-of-two scaled float: the reference uses the int
            # PerChannelPoTScaling8bit mixin layered on a float quant. The
            # generic builder path reproduces it; any attribute that resolves
            # differently from the reference is overridden here.
            "kwargs": {},},
        "layer_kwargs": {},},
    # ----------------------------------------------------------------------
    # float_fnuz / float_scale: WEIGHT_QUANT_MAP['float_fnuz']['float_scale'].
    # ----------------------------------------------------------------------
    "float_fnuz_per_tensor_sym": {
        "ref": Fp8e4m3FNUZWeightPerTensorFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.FNUZ,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.TENSOR,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},
    "float_fnuz_per_channel_sym": {
        "ref": Fp8e4m3FNUZWeightPerChannelFloat,
        "builder_args": {
            "quant_type": QuantType.FP,
            "quant_param_type": QuantParamType.SYM,
            "float_format": FloatFormat.FNUZ,
            "float_quant_format": "e4m3",
            "scaling_per_output_type": ScalingPerOutputType.CHANNEL,
            "restrict_scaling_type": RestrictValueType.FP,
            "scaling_min_val": SCALING_MIN_VAL,
            "kwargs": {},},
        "layer_kwargs": {},},}


def _make_quant_linear(weight_quant, **layer_kwargs):
    # NOTE: return_quant_tensor must be False here. With weight-only
    # quantization (no input_quant), the layer input stays a plain Tensor, so
    # the layer cannot emit a QuantTensor output and would otherwise raise
    # "QuantLayer is not correctly configured". The quantized weight is still
    # compared directly via ``quant_weight()``, which returns a QuantTensor
    # regardless of this flag.
    linear = QuantLinear(
        IN_FEATURES,
        OUT_FEATURES,
        bias=False,
        weight_quant=weight_quant,
        return_quant_tensor=False,
        **layer_kwargs)
    return linear


def _module_hierarchy(model):
    """Return an ordered, comparable description of the module hierarchy.

    Each entry is a (name, fully-qualified-type) pair, so two models match
    1-to-1 only if they have exactly the same submodules, in the same order,
    of the same types.
    """
    hierarchy = []
    for name, module in model.named_modules():
        type_ = type(module)
        hierarchy.append((name, f"{type_.__module__}.{type_.__qualname__}"))
    return hierarchy


@pytest.mark.parametrize("spec_name", list(BUILDER_SPECS.keys()))
def test_builder_weight_quant_matches_reference(spec_name):
    spec = BUILDER_SPECS[spec_name]
    ref_quant = spec["ref"]
    layer_kwargs = spec["layer_kwargs"]

    # Local-loss param methods (MSE, HQO) rely on Python control flow during
    # the optimization and require JIT to be disabled.
    local_loss_methods = (ParamMethod.MSE, ParamMethod.HQO)
    param_methods = (
        spec["builder_args"].get("scaling_param_method"),
        spec["builder_args"].get("zero_point_param_method"))
    if config.JIT_ENABLED and any(m in local_loss_methods for m in param_methods):
        pytest.skip(reason="Local loss param methods (MSE, HQO) require JIT to be disabled")

    # Some WEIGHT_QUANT_MAP entries are broken in the reference brevitas
    # quantizer itself; mark those as expected failures.
    if "xfail" in spec:
        pytest.xfail(reason=spec["xfail"])

    # Reference layer built directly from the WEIGHT_QUANT_MAP leaf class.
    ref_linear = _make_quant_linear(ref_quant, **layer_kwargs)

    # Builder layer built from the generic QuantizerBuilder.
    builder = build_weight_quantizer(**spec["builder_args"])
    builder_quant = builder.build_quant_injector()
    builder_linear = _make_quant_linear(builder_quant, **layer_kwargs)

    # 1) Module hierarchy must match 1-to-1. Checked before syncing weights so a
    # structural mismatch is reported as a clear hierarchy diff rather than an
    # opaque "Missing key(s) in state_dict" error.
    # if _module_hierarchy(ref_linear) != _module_hierarchy(builder_linear):
    #     breakpoint()
    assert _module_hierarchy(ref_linear) == _module_hierarchy(builder_linear)

    # Make both layers carry identical float weights so the only difference that
    # could appear is in the quantization path itself. We copy only the float
    # weight (not the full state_dict): for MSE / PARAMETER_FROM_STATS scaling
    # the learned scale parameter (scaling_impl.value) is excluded from
    # state_dict() until it has been initialized, so a strict load_state_dict
    # would spuriously fail with a missing key.
    builder_linear.weight.data.copy_(ref_linear.weight.data)

    ref_linear.eval()
    builder_linear.eval()

    # Mock forward pass to trigger lazy initialization of any parameter-based
    # scaling (e.g. MSE / PARAMETER_FROM_STATS). After this, scaling_impl.value
    # is initialized from the (now identical) weights on both layers, so the
    # learned scales are directly comparable.
    mock_input = torch.randn(1, IN_FEATURES)
    ref_linear(mock_input)
    builder_linear(mock_input)

    # 2) The quantized weight tensors themselves must match exactly.
    ref_weight = ref_linear.quant_weight()
    builder_weight = builder_linear.quant_weight()
    assert torch.equal(ref_weight.value, builder_weight.value)
    assert torch.equal(ref_weight.scale, builder_weight.scale)
    assert (ref_weight.zero_point is None) == (builder_weight.zero_point is None)
    if ref_weight.zero_point is not None:
        assert torch.equal(ref_weight.zero_point, builder_weight.zero_point)
    # Int quant tensors expose `bit_width`; float quant tensors instead expose
    # `exponent_bit_width` / `mantissa_bit_width`.
    if hasattr(ref_weight, "bit_width"):
        assert torch.equal(ref_weight.bit_width, builder_weight.bit_width)
    else:
        assert torch.equal(ref_weight.exponent_bit_width, builder_weight.exponent_bit_width)
        assert torch.equal(ref_weight.mantissa_bit_width, builder_weight.mantissa_bit_width)

    # 3) Quantized layer output tensors must match exactly. With
    # return_quant_tensor=False the layers return plain Tensors.
    x = torch.randn(1, IN_FEATURES)
    ref_out = ref_linear(x)
    builder_out = builder_linear(x)
    assert torch.equal(ref_out, builder_out)
