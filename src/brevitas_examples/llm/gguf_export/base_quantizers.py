# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Brevitas weight quantizers that reproduce the GGUF/GGML block layouts.

Each quantizer below is calibrated by Brevitas as usual, but its scale/zero-point
structure is chosen so that the resulting quant parameters can be packed by the
encoders in :mod:`brevitas_examples.llm.gguf_export.quant` without any loss:

* ``GGUFQ8_0WeightQuant`` -- blocks of 32, signed 8-bit, single fp16 group scale.
* ``GGUFQ6_KWeightQuant`` -- super-blocks of 256 (16 sub-blocks of 16), signed
  6-bit, symmetric, with 8-bit signed sub-block scales each scaled by a single
  per-super-block fp16 ``d`` (nested "double" quantization, no min). Used for the
  high-impact ``token_embd`` / ``output`` tensors.
* ``GGUFQ5_KWeightQuant`` -- like Q4_K but with 5-bit codes; same nested structure.
* ``GGUFQ4_KWeightQuant`` -- super-blocks of 256 (8 sub-blocks of 32), unsigned
  4-bit, with 6-bit sub-block scales/mins each scaled by a per-super-block fp16
  ``d`` / ``dmin`` (nested "double" quantization).
* ``GGUFQ4_1WeightQuant`` -- blocks of 32, unsigned 4-bit, fp16 scale + fp16 min.
* ``GGUFQ4_0WeightQuant`` -- blocks of 32, signed 4-bit, single fp16 group scale.
* ``GGUFQ3_KWeightQuant`` -- super-blocks of 256 (16 sub-blocks of 16), signed
  3-bit, symmetric, with 6-bit signed sub-block scales each scaled by a single
  per-super-block fp16 ``d`` (nested "double" quantization, no min).
* ``GGUFQ2_KWeightQuant`` -- super-blocks of 256 (16 sub-blocks of 16), unsigned
  2-bit, asymmetric, with 4-bit sub-block scales/mins each scaled by a
  per-super-block fp16 ``d`` / ``dmin`` (nested "double" quantization).

Each quantizer inherits from :class:`_GGUFBaseQuant`, uses
:class:`GGUFGroupwiseWeightQuantProxyFromInjector` as its ``proxy_class``, and
declares a ``gguf_qtype`` class attribute.
"""

from dependencies import this
from dependencies import value
import gguf
import torch

from brevitas.core.function_wrapper import CeilSte
from brevitas.core.function_wrapper import FloorSte
from brevitas.core.function_wrapper.shape import StatsInputViewShapeImpl
from brevitas.core.quant.int import RescalingIntQuant
from brevitas.core.restrict_val import QuantRestrictValue
from brevitas.core.restrict_val import RoundSte
from brevitas.core.scaling import RoundMidMaxSte
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.core.zero_point import _ScaleShiftQuantZeroPoint
from brevitas.export.inference.handler import GroupwiseIntWeightInferenceHandler
from brevitas.export.inference.manager import InferenceManager
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.inject.enum import StatsOp
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.quant.base import ExtendedInjector
from brevitas.quant.base import FloatRestrictValue
from brevitas.quant.base import MSEAsymmetricScale
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat

# GGML block geometry. A K-quant super-block spans QK_K elements; the smaller
# integer block sizes (Q4_0/Q4_1/Q8_0) span QK.
QK = 32
QK_K = 256

# K-quant sub-block geometry and nested scale bit-widths (high to low weight precision).
Q6_K_GROUP_SIZE = 16  # 16 sub-blocks of 16 per 256-element super-block
Q6_K_SUB_SCALE_BIT_WIDTH = 8

Q5_K_GROUP_SIZE = 32  # 8 sub-blocks of 32 per 256-element super-block
Q5_K_SUB_SCALE_BIT_WIDTH = 6
Q5_K_SUB_ZP_BIT_WIDTH = 6

Q4_K_GROUP_SIZE = 32
Q4_K_SUB_SCALE_BIT_WIDTH = 6
Q4_K_SUB_ZP_BIT_WIDTH = 6

Q3_K_GROUP_SIZE = 16
Q3_K_SUB_SCALE_BIT_WIDTH = 6

Q2_K_GROUP_SIZE = 16
Q2_K_SUB_SCALE_BIT_WIDTH = 4
Q2_K_SUB_ZP_BIT_WIDTH = 4


class GGUFGroupwiseWeightQuantProxyFromInjector(GroupwiseWeightQuantProxyFromInjector):
    """Groupwise weight proxy for GGUF quantizers; carries the declared qtype"""

    @property
    def gguf_qtype(self):
        return self.quant_injector.gguf_qtype


class _GGUFGroupwiseIntWeightInferenceHandler(GroupwiseIntWeightInferenceHandler):
    handled_layer = GGUFGroupwiseWeightQuantProxyFromInjector


# TODO: temporary workaround to tag/export GGUF qtypes; should revist
InferenceManager.handlers.append(_GGUFGroupwiseIntWeightInferenceHandler)


class _GGUFBaseQuant(ExtendedInjector):
    """Marker base for GGUF weight quantizers that declare a `gguf_qtype` tag.

    Carries the defaults shared by every GGUF quantizer: the GGUF-aware proxy,
    per-group scaling, and full-range (non-narrow) integer codes.
    """
    proxy_class = GGUFGroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    narrow_range = False


class GGUFQ8_0WeightQuant(_GGUFBaseQuant, Int8WeightPerChannelFloat):
    """Signed symmetric 8-bit group quantizer with one fp16 scale per 32-element block."""
    gguf_qtype = gguf.GGMLQuantizationType.Q8_0
    group_size = QK
    bit_width = 8


class GGUFQ4_1WeightQuant(_GGUFBaseQuant, ShiftedUint8WeightPerChannelFloat):
    """Asymmetric unsigned 4-bit group quantizer with fp16 scale + fp16 min per block.

    The packed min stored by GGUF Q4_1 is ``-zero_point * scale``.
    """
    gguf_qtype = gguf.GGMLQuantizationType.Q4_1
    group_size = QK
    bit_width = 4


class GGUFQ4_0WeightQuant(_GGUFBaseQuant, Int8WeightPerChannelFloat):
    """Signed symmetric 4-bit group quantizer with one fp16 scale per 32-element block."""
    gguf_qtype = gguf.GGMLQuantizationType.Q4_0
    group_size = QK
    bit_width = 4


# ---------------------------------------------------------------------------
# K-quants: nested ("double") quantization of the sub-block scales (and mins).
#
# The per-group scale (and zero-point) are themselves quantized against a
# per-super-block fp16 factor via QuantRestrictValue / _ScaleShiftQuantZeroPoint.
# convert.py reads those exact modules off the calibrated weight_quant to recover
# d_scale / d_wmin_m.
#
#   Q6_K -> 16 sub-blocks of 16, symmetric: 8-bit signed scales, single fp16
#           super-block d (no min / zero-point).
#   Q5_K -> like Q4_K but 5-bit weight codes.
#   Q4_K -> 8 sub-blocks of 32, asymmetric: 6-bit unsigned scales + 6-bit
#           unsigned mins, fp16 super-block d / dmin.
#   Q3_K -> 16 sub-blocks of 16, symmetric: 6-bit signed scales, single fp16
#           super-block d (no min / zero-point).
#   Q2_K -> 16 sub-blocks of 16, asymmetric: 4-bit unsigned scales + 4-bit
#           unsigned mins, fp16 super-block d / dmin.
# ---------------------------------------------------------------------------


class _GGUFBaseKQuant(_GGUFBaseQuant):
    """Common base for the K-quant weight quantizers."""
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


class _GGUFShiftedBaseKQuant(_GGUFBaseKQuant):
    """Common base for asymmetric K-quants with nested scale + zero-point (min)."""
    scale_shift_zero_point_impl = _ScaleShiftQuantZeroPoint

    @value
    def zp_int_quant():
        return this.zp_quant.rescaling_int_quant

    @value
    def zero_point_dequantized_shape(scaling_per_output_type, zero_point_shape):
        if scaling_per_output_type == ScalingPerOutputType.GROUP:
            return zero_point_shape
        return None


class _GGUFSignedBaseKQuant(_GGUFBaseKQuant):
    """Base quantizer for K-quants with signed scales"""
    signed = True
    restrict_scaling_type = RestrictValueType.SIGNED_FP
    scaling_stats_op = StatsOp.SIGNED_MAX


_SCALE_ROUNDING_FUNC_DICT = {
    'ceil': CeilSte, 'floor': FloorSte, 'round': RoundSte, 'midmax': RoundMidMaxSte}


class _GGUFKQuantShapeMixin(ExtendedInjector):
    """Common base for every nested K-quant scale/zero-point sub-injector.
    """
    narrow_range = False  # scale/zp quantization is always full-range too

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


class _GGUFKQuantScalingMixin(_GGUFKQuantShapeMixin):
    """Common wiring for a nested K-quant scale sub-injector (the scale-of-scale).

    Forwards ``module`` from the parent quantizer and ties ``upstream_shape`` to its
    per-sub-block ``scaling_shape``.
    """
    module = (this << 1).module
    rescaling_int_quant = RescalingIntQuant
    scaling_per_output_type = ScalingPerOutputType.GROUP
    upstream_shape = (this << 1).scaling_shape
    scale_rounding_func_type = 'midmax'

    @value
    def mantissa_bit_width_impl():
        # TODO: this is a temporary workaround to use midmax with integers
        return this.bit_width_impl

    @value
    def tracked_parameter_list(upstream_shape):
        return [torch.empty(upstream_shape)]

    @value
    def float_to_int_impl(scale_rounding_func_type):
        return _SCALE_ROUNDING_FUNC_DICT[scale_rounding_func_type]


class _GGUFKQuantZPMixin(_GGUFKQuantShapeMixin):
    """Common wiring for a nested K-quant zero-point/min sub-injector (the min-of-min).

    Forwards ``module`` from the parent quantizer and ties ``upstream_shape`` to its
    per-sub-block ``zero_point_shape``.
    """
    module = (this << 1).module
    rescaling_int_quant = RescalingIntQuant
    restrict_threshold_impl = FloatRestrictValue
    scaling_per_output_type = ScalingPerOutputType.GROUP
    upstream_shape = (this << 1).zero_point_shape
    scale_rounding_func_type = 'round'

    @value
    def tracked_parameter_list(upstream_shape):
        return [torch.empty(upstream_shape)]

    @value
    def float_to_int_impl(scale_rounding_func_type):
        return _SCALE_ROUNDING_FUNC_DICT[scale_rounding_func_type]


class _GGUFQ6_KScalingInt(_GGUFKQuantScalingMixin, Int8WeightPerChannelFloat):
    """8-bit signed quantizer for the per-sub-block scales (the scale-of-scale).

    Q6_K stores 16 int8 sub-block scales per super-block, each scaled by a single
    fp16 super-block factor ``d``. llama.cpp anchors the max-magnitude sub-scale
    to -128 (full signed range), so we use a *signed* scale factor here
    (SIGNED_FP restrict + SIGNED_MAX stats) to match its quantization exactly.
    """
    bit_width = Q6_K_SUB_SCALE_BIT_WIDTH
    group_size = QK_K // Q6_K_GROUP_SIZE
    signed = True
    restrict_scaling_type = RestrictValueType.SIGNED_FP
    scaling_stats_op = StatsOp.SIGNED_MAX


class GGUFQ6_KWeightQuant(_GGUFSignedBaseKQuant, Int8WeightPerChannelFloat):
    """Signed symmetric 6-bit Q6_K super-block quantizer with nested scales.

    256-element super-blocks split into 16 sub-blocks of 16. Each sub-block has an
    8-bit signed scale, quantized against a single per-super-block fp16 factor
    ``d`` (no min / zero-point). Used for the high-impact ``token_embd`` /
    ``output`` tensors under the GGUF Q4_K standard.
    """
    gguf_qtype = gguf.GGMLQuantizationType.Q6_K
    group_size = Q6_K_GROUP_SIZE
    bit_width = 6
    scaling_quant = _GGUFQ6_KScalingInt


class _GGUFQ5_KScalingInt(_GGUFKQuantScalingMixin, Int8WeightPerTensorFloat):
    """6-bit unsigned quantizer for the per-sub-block scales (the scale-of-scale)."""
    bit_width = Q4_K_SUB_SCALE_BIT_WIDTH
    group_size = QK_K // Q4_K_GROUP_SIZE
    signed = False


class _GGUFQ5_KZPInt(_GGUFKQuantZPMixin, Int8WeightPerTensorFloat):
    """6-bit unsigned quantizer for the per-sub-block mins (the min-of-min)."""
    bit_width = Q4_K_SUB_ZP_BIT_WIDTH
    group_size = QK_K // Q4_K_GROUP_SIZE
    signed = False


class GGUFQ5_KWeightQuant(_GGUFShiftedBaseKQuant, ShiftedUint8WeightPerTensorFloat):
    """Asymmetric unsigned 5-bit Q5_K super-block quantizer with nested scales/mins.

    Identical to :class:`GGUFQ4_KWeightQuant` (same 8 sub-blocks of 32, 6-bit nested
    scales/mins, fp16 ``d`` / ``dmin``) except the weight codes use 5 bits ([0, 31])
    instead of 4. Reuses the same nested scale/zero-point sub-injectors.
    """
    gguf_qtype = gguf.GGMLQuantizationType.Q5_K
    group_size = Q5_K_GROUP_SIZE
    bit_width = 5
    scaling_quant = _GGUFQ5_KScalingInt
    zp_quant = _GGUFQ5_KZPInt


class GGUFQ4_KWeightQuant(GGUFQ5_KWeightQuant):
    """Asymmetric unsigned 4-bit Q4_K super-block quantizer with nested scales/mins.

    256-element super-blocks split into 8 sub-blocks of 32. Each sub-block has a
    6-bit scale and a 6-bit min, both quantized against a per-super-block fp16
    factor (``d`` / ``dmin``).
    """
    gguf_qtype = gguf.GGMLQuantizationType.Q4_K
    group_size = Q4_K_GROUP_SIZE
    bit_width = 4


class _GGUFQ3_KScalingInt(_GGUFKQuantScalingMixin, Int8WeightPerChannelFloat):
    """6-bit *signed* quantizer for the per-sub-block scales (the scale-of-scale).

    Q3_K stores 16 signed 6-bit sub-block scales per super-block, each scaled by a
    single fp16 super-block factor ``d`` -- same 16-sub-block group size as Q6_K's
    nested scale, just narrower (6 bits, not 8). On disk the signed code is stored
    biased by +32 as an unsigned value (``ggml-quants.c``'s ``sc + 32``, range
    [0, 63]); that bias is applied only at pack time in quant.py's ``_q3_k_pack``,
    so this quantizer itself stays signed/symmetric like Q6_K's.
    """
    bit_width = Q3_K_SUB_SCALE_BIT_WIDTH
    group_size = QK_K // Q3_K_GROUP_SIZE  # 16 sub-blocks share one super-block factor
    signed = True
    restrict_scaling_type = RestrictValueType.SIGNED_FP
    scaling_stats_op = StatsOp.SIGNED_MAX


class GGUFQ3_KWeightQuant(_GGUFSignedBaseKQuant, Int8WeightPerChannelFloat):
    """Signed symmetric 3-bit Q3_K super-block quantizer with nested scales.

    256-element super-blocks split into 16 sub-blocks of 16. Each sub-block has a
    6-bit signed scale (stored biased by +32 as an unsigned value on disk),
    quantized against a single per-super-block fp16 factor ``d`` (no min /
    zero-point).
    """
    gguf_qtype = gguf.GGMLQuantizationType.Q3_K
    group_size = Q3_K_GROUP_SIZE
    bit_width = 3
    scaling_quant = _GGUFQ3_KScalingInt


class _GGUFQ2_KScalingInt(_GGUFKQuantScalingMixin, Int8WeightPerTensorFloat):
    """4-bit unsigned quantizer for the per-sub-block scales (the scale-of-scale).

    Q2_K uses 16 sub-blocks of 16 (not Q4_K's 8 of 32), each with a 4-bit (not
    6-bit) scale code, packed on disk as a plain nibble (see quant.py's
    ``_q2_k_pack``) rather than Q4_K's 6-bit cross-byte interleave.
    """
    bit_width = Q2_K_SUB_SCALE_BIT_WIDTH
    group_size = QK_K // Q2_K_GROUP_SIZE
    signed = False


class _GGUFQ2_KZPInt(_GGUFKQuantZPMixin, Int8WeightPerTensorFloat):
    """4-bit unsigned quantizer for the per-sub-block mins (the min-of-min)."""
    bit_width = Q2_K_SUB_ZP_BIT_WIDTH
    group_size = QK_K // Q2_K_GROUP_SIZE
    signed = False
    scale_rounding_func_type = 'floor'


class GGUFQ2_KWeightQuant(_GGUFShiftedBaseKQuant,
                          MSEAsymmetricScale,
                          ShiftedUint8WeightPerTensorFloat):
    """Asymmetric unsigned 2-bit Q2_K super-block quantizer with nested scales/mins.

    256-element super-blocks split into 16 sub-blocks of 16. Each sub-block has a
    4-bit scale and a 4-bit min, both quantized against a per-super-block fp16
    factor (``d`` / ``dmin``). Unlike Q4_K/Q5_K's 6-bit interleaved packing, the
    4+4 bit scale/min pair fits exactly one byte per sub-block (plain nibble pack).
    """
    gguf_qtype = gguf.GGMLQuantizationType.Q2_K
    group_size = Q2_K_GROUP_SIZE
    bit_width = 2
    scaling_quant = _GGUFQ2_KScalingInt
    zp_quant = _GGUFQ2_KZPInt
    # MSEAsymmetricScale sets scaling_stats_input_view_shape_impl = Identity
    zero_point_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_SUBCHANNEL_BLOCK
