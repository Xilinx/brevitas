# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Brevitas weight quantizers that reproduce the GGUF/GGML block layouts.

Each quantizer below is calibrated by Brevitas as usual, but its scale/zero-point
structure is chosen so that the resulting quant parameters can be packed by the
encoders in :mod:`brevitas_examples.llm.gguf_export.quant` without any loss:

* ``GGUFQ4_0WeightQuant`` -- blocks of 32, signed 4-bit, single fp16 group scale.
* ``GGUFQ4_1WeightQuant`` -- blocks of 32, unsigned 4-bit, fp16 scale + fp16 min.
* ``GGUFQ4_KWeightQuant`` -- super-blocks of 256 (8 sub-blocks of 32), unsigned
  4-bit, with 6-bit sub-block scales/mins each scaled by a per-super-block fp16
  ``d`` / ``dmin`` (nested "double" quantization). This is the quantizer that
  :func:`...convert.ModelBase.quantize` inspects for ``QuantRestrictValue`` /
  ``_ScaleShiftQuantZeroPoint`` modules.
* ``GGUFQ6_KWeightQuant`` -- super-blocks of 256 (16 sub-blocks of 16), signed
  6-bit, symmetric, with 8-bit signed sub-block scales each scaled by a single
  per-super-block fp16 ``d`` (nested "double" quantization, no min). Used for the
  high-impact ``token_embd`` / ``output`` tensors.
* ``GGUFQ8_0WeightQuant`` -- blocks of 32, signed 8-bit, single fp16 group scale.
"""

from dependencies import this
from dependencies import value
import torch

from brevitas.core.quant.int import RescalingIntQuant
from brevitas.core.restrict_val import QuantRestrictValue
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.core.zero_point import _ScaleShiftQuantZeroPoint
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.inject.enum import StatsOp
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.quant.base import ExtendedInjector
from brevitas.quant.base import FloatRestrictValue
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat

# GGML block geometry. A K-quant super-block spans QK_K elements; the smaller
# integer block sizes (Q4_0/Q4_1/Q8_0) span QK.
QK = 32
QK_K = 256
# Sub-block group sizes inside a K-quant super-block (block_q4_K / block_q6_K).
Q4_K_GROUP_SIZE = 32  # 8 sub-blocks of 32 per 256-element super-block
Q6_K_GROUP_SIZE = 16  # 16 sub-blocks of 16 per 256-element super-block
# Bit-widths used for the nested (scale-of-scale / min-of-min) quantization.
#  Q4_K: 6-bit *unsigned* sub-block scales and mins, fp16 super-block d / dmin.
#  Q6_K: 8-bit *signed* sub-block scales, single fp16 super-block d (symmetric).
Q4_K_SUB_SCALE_BIT_WIDTH = 6
Q4_K_SUB_ZP_BIT_WIDTH = 6
Q6_K_SUB_SCALE_BIT_WIDTH = 8
# Q5_K shares Q4_K's super-block structure (8 sub-blocks of 32, 6-bit unsigned
# nested scales/mins, fp16 d/dmin); only the weight code bit-width differs (5 vs 4).
Q5_K_GROUP_SIZE = 32
Q5_K_SUB_SCALE_BIT_WIDTH = 6
Q5_K_SUB_ZP_BIT_WIDTH = 6


class GGUFQ4_0WeightQuant(Int8WeightPerChannelFloat):
    """Signed symmetric 4-bit group quantizer with one fp16 scale per 32-element block."""
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = QK
    bit_width = 4
    narrow_range = False


class GGUFQ8_0WeightQuant(Int8WeightPerChannelFloat):
    """Signed symmetric 8-bit group quantizer with one fp16 scale per 32-element block."""
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = QK
    bit_width = 8
    narrow_range = False


class GGUFQ4_1WeightQuant(ShiftedUint8WeightPerChannelFloat):
    """Asymmetric unsigned 4-bit group quantizer with fp16 scale + fp16 min per block.

    The packed min stored by GGUF Q4_1 is ``-zero_point * scale``.
    """
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = QK
    bit_width = 4


# ---------------------------------------------------------------------------
# K-quants: nested ("double") quantization of the sub-block scales (and mins).
#
# This mirrors the canonical pattern exercised in
# tests/brevitas/core/test_scaling_quant.py: the per-group scale (and zero-point)
# are themselves quantized against a per-super-block fp16 factor via
# QuantRestrictValue / _ScaleShiftQuantZeroPoint. convert.py reads those exact
# modules off the calibrated weight_quant to recover d_scale / d_wmin_m.
#
#   Q4_K -> 8 sub-blocks of 32, asymmetric: 6-bit unsigned scales + 6-bit
#           unsigned mins, fp16 super-block d / dmin.
#   Q6_K -> 16 sub-blocks of 16, symmetric: 8-bit signed scales, single fp16
#           super-block d (no min / zero-point).
# ---------------------------------------------------------------------------


class _GGUFKQuantShapeMixin(ExtendedInjector):

    @value
    def scaling_shape(
            scaling_per_output_type,
            scaling_per_output_channel_shape,
            expanded_groupwise_shape,
            group_dim):
        if scaling_per_output_type == ScalingPerOutputType.TENSOR:
            return SCALAR_SHAPE
        elif scaling_per_output_type == ScalingPerOutputType.CHANNEL:
            return scaling_per_output_channel_shape
        # GROUP: like expanded_groupwise_shape but with 1 in position group_dim + 1.
        assert expanded_groupwise_shape is not None, "Per Group scaling not correctly configured"
        assert group_dim is not None, "Per Group scaling not correctly configured"
        size = list(expanded_groupwise_shape)
        size[group_dim + 1] = 1
        return tuple(size)


class _GGUFQ4KScalingInt(Int8WeightPerTensorFloat, _GGUFKQuantShapeMixin):
    """6-bit unsigned quantizer for the per-sub-block scales (the scale-of-scale)."""
    module = (this << 1).module
    rescaling_int_quant = RescalingIntQuant
    bit_width = Q4_K_SUB_SCALE_BIT_WIDTH
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = QK_K // Q4_K_GROUP_SIZE  # 8 sub-blocks share one super-block factor
    upstream_shape = (this << 1).scaling_shape
    signed = False

    @value
    def tracked_parameter_list(upstream_shape):
        return [torch.empty(upstream_shape)]


class _GGUFQ4KZPInt(Int8WeightPerTensorFloat, _GGUFKQuantShapeMixin):
    """6-bit unsigned quantizer for the per-sub-block mins (the min-of-min)."""
    module = (this << 1).module
    rescaling_int_quant = RescalingIntQuant
    restrict_threshold_impl = FloatRestrictValue
    bit_width = Q4_K_SUB_ZP_BIT_WIDTH
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = QK_K // Q4_K_GROUP_SIZE
    upstream_shape = (this << 1).zero_point_shape
    signed = False

    @value
    def tracked_parameter_list(upstream_shape):
        return [torch.empty(upstream_shape)]


class GGUFQ4_KWeightQuant(ShiftedUint8WeightPerTensorFloat):
    """Asymmetric unsigned 4-bit Q4_K super-block quantizer with nested scales/mins.

    256-element super-blocks split into 8 sub-blocks of 32. Each sub-block has a
    6-bit scale and a 6-bit min, both quantized against a per-super-block fp16
    factor (``d`` / ``dmin``).
    """
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = Q4_K_GROUP_SIZE
    bit_width = 4

    scaling_quant = _GGUFQ4KScalingInt
    zp_quant = _GGUFQ4KZPInt
    restrict_scaling_impl = QuantRestrictValue
    restrict_threshold_impl = FloatRestrictValue
    scale_shift_zero_point_impl = _ScaleShiftQuantZeroPoint

    @value
    def restrict_value_float_to_int_impl():
        return this.scaling_quant.rescaling_int_quant

    @value
    def zp_int_quant():
        return this.zp_quant.rescaling_int_quant

    @value
    def scale_dequantized_shape(scaling_per_output_type, scaling_shape):
        if scaling_per_output_type == ScalingPerOutputType.GROUP:
            return scaling_shape
        return None

    @value
    def zero_point_dequantized_shape(scaling_per_output_type, zero_point_shape):
        if scaling_per_output_type == ScalingPerOutputType.GROUP:
            return zero_point_shape
        return None


class GGUFQ5_KWeightQuant(GGUFQ4_KWeightQuant):
    """Asymmetric unsigned 5-bit Q5_K super-block quantizer with nested scales/mins.

    Identical to :class:`GGUFQ4_KWeightQuant` (same 8 sub-blocks of 32, 6-bit nested
    scales/mins, fp16 ``d`` / ``dmin``) except the weight codes use 5 bits ([0, 31])
    instead of 4. Reuses the same nested scale/zero-point sub-injectors.
    """
    group_size = Q5_K_GROUP_SIZE
    bit_width = 5


class _GGUFQ6KScalingInt(Int8WeightPerChannelFloat, _GGUFKQuantShapeMixin):
    """8-bit *signed* quantizer for the per-sub-block scales (the scale-of-scale).

    Q6_K stores 16 int8 sub-block scales per super-block, each scaled by a single
    fp16 super-block factor ``d``. llama.cpp anchors the max-magnitude sub-scale
    to -128 (full signed range), so we use a *signed* scale factor here
    (SIGNED_FP restrict + SIGNED_MAX stats) to match its quantization exactly.
    """
    module = (this << 1).module
    rescaling_int_quant = RescalingIntQuant
    bit_width = Q6_K_SUB_SCALE_BIT_WIDTH
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = QK_K // Q6_K_GROUP_SIZE  # 16 sub-blocks share one super-block factor
    upstream_shape = (this << 1).scaling_shape
    signed = True
    narrow_range = False
    # Signed scale factor: anchor the max-magnitude sub-scale to the signed edge.
    restrict_scaling_type = RestrictValueType.SIGNED_FP
    scaling_stats_op = StatsOp.SIGNED_MAX

    @value
    def tracked_parameter_list(upstream_shape):
        return [torch.empty(upstream_shape)]


class GGUFQ6_KWeightQuant(Int8WeightPerChannelFloat):
    """Signed symmetric 6-bit Q6_K super-block quantizer with nested scales.

    256-element super-blocks split into 16 sub-blocks of 16. Each sub-block has an
    8-bit signed scale, quantized against a single per-super-block fp16 factor
    ``d`` (no min / zero-point). Used for the high-impact ``token_embd`` /
    ``output`` tensors under the GGUF Q4_K standard.
    """
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = Q6_K_GROUP_SIZE
    bit_width = 6
    narrow_range = False

    scaling_quant = _GGUFQ6KScalingInt
    restrict_scaling_impl = QuantRestrictValue
    restrict_threshold_impl = FloatRestrictValue

    @value
    def restrict_value_float_to_int_impl():
        return this.scaling_quant.rescaling_int_quant

    @value
    def scale_dequantized_shape(scaling_per_output_type, scaling_shape):
        if scaling_per_output_type == ScalingPerOutputType.GROUP:
            return scaling_shape
        return None
