# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Brevitas weight quantizers that reproduce the GGUF/GGML block layouts.

Each quantizer below is calibrated by Brevitas as usual, but its scale/zero-point
structure is chosen so that the resulting quant parameters can be packed by the
encoders in :mod:`brevitas_examples.llm.gguf_export.quant` without any loss:

* ``GGUFQ8_0WeightQuant`` -- blocks of 32, signed 8-bit, single fp16 group scale.
* ``GGUFQ4_1WeightQuant`` -- blocks of 32, unsigned 4-bit, fp16 scale + fp16 min.
* ``GGUFQ4_0WeightQuant`` -- blocks of 32, signed 4-bit, single fp16 group scale.

* ``GGUFQ6_KWeightQuant`` -- super-blocks of 256 (16 sub-blocks of 16), signed
  6-bit, symmetric, with 8-bit signed sub-block scales each scaled by a single
  per-super-block fp16 ``d`` (nested "double" quantization, no min).
* ``GGUFQ5_KWeightQuant`` -- super-blocks of 256 (8 sub-blocks of 32), unsigned
  5-bit, with 6-bit sub-block scales/mins each scaled by a per-super-block fp16
  ``d`` / ``dmin`` (nested "double" quantization).
* ``GGUFQ4_KWeightQuant`` -- like Q5_K but with 4-bit codes; same nested structure.
* ``GGUFQ3_KWeightQuant`` -- super-blocks of 256 (16 sub-blocks of 16), signed
  3-bit, symmetric, with 6-bit signed sub-block scales each scaled by a single
  per-super-block fp16 ``d`` (nested "double" quantization, no min).
* ``GGUFQ2_KWeightQuant`` -- super-blocks of 256 (16 sub-blocks of 16), unsigned
  2-bit, asymmetric, with 4-bit sub-block scales/mins each scaled by a
  per-super-block fp16 ``d`` / ``dmin`` (nested "double" quantization).

Each quantizer uses `GGUFGroupwiseWeightQuantProxyFromInjector` as its `proxy_class`,
and declares a `gguf_qtype` class attribute.
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
from brevitas.quant.base import MSESymmetricScale
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat

# GGML block geometry: K-quant super-block spans QK_K elements; Q4_0/Q4_1/Q8_0 span QK.
QK = 32
QK_K = 256

# K-quant sub-block geometry and nested scale bit widths
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


class _GGUFBaseQuantMixin(ExtendedInjector):
    """Base GGUF quantizer mixin. Carries the defaults shared by every GGUF quantizer:
    the GGUF-aware proxy, per-group scaling, and full-range (non-narrow) integer codes."""
    proxy_class = GGUFGroupwiseWeightQuantProxyFromInjector
    scaling_per_output_type = ScalingPerOutputType.GROUP
    narrow_range = False


class GGUFQ8_0WeightQuant(_GGUFBaseQuantMixin, Int8WeightPerChannelFloat):
    """Signed symmetric 8-bit group quantizer with one fp16 scale per 32-element block."""
    gguf_qtype = gguf.GGMLQuantizationType.Q8_0
    group_size = QK
    bit_width = 8


class GGUFQ4_1WeightQuant(_GGUFBaseQuantMixin, ShiftedUint8WeightPerChannelFloat):
    """Asymmetric unsigned 4-bit group quantizer with fp16 scale + fp16 min per block."""
    gguf_qtype = gguf.GGMLQuantizationType.Q4_1
    group_size = QK
    bit_width = 4


class GGUFQ4_0WeightQuant(_GGUFBaseQuantMixin, Int8WeightPerChannelFloat):
    """Signed symmetric 4-bit group quantizer with one fp16 scale per 32-element block."""
    gguf_qtype = gguf.GGMLQuantizationType.Q4_0
    group_size = QK
    bit_width = 4


# ---------------------------------------------------------------------------
# K-quants: nested ("double") quantization of the sub-block scales (and mins).
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


class __GGUFBaseKQuantMixin(_GGUFBaseQuantMixin):
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


class _GGUFShiftedBaseKQuantMixin(__GGUFBaseKQuantMixin, MSEAsymmetricScale):
    """Base quantizer for asymmetric K-quants with nested scale + zero-point (min)."""
    scale_shift_zero_point_impl = _ScaleShiftQuantZeroPoint
    restrict_scale_positive = True
    # MSEAsymmetricScale sets scaling_stats_input_view_shape_impl = Identity; pin it explicitly
    zero_point_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_SUBCHANNEL_BLOCK

    @value
    def zp_int_quant():
        return this.zp_quant.rescaling_int_quant

    @value
    def zero_point_dequantized_shape(scaling_per_output_type, zero_point_shape):
        if scaling_per_output_type == ScalingPerOutputType.GROUP:
            return zero_point_shape
        return None


class _GGUFSignedBaseKQuantMixin(__GGUFBaseKQuantMixin, MSESymmetricScale):
    """Base quantizer for signed symmetric K-quants with nested scales."""
    signed = True
    restrict_scaling_type = RestrictValueType.SIGNED_FP
    scaling_stats_op = StatsOp.SIGNED_MAX
    restrict_scale_positive = False


class __GGUFKQuantScaleZPMixin(ExtendedInjector):
    """Common base for every nested K-quant scale/zero-point sub-injector."""
    narrow_range = False  # scale/zp quantization is always full-range too
    rescaling_int_quant = RescalingIntQuant
    scaling_per_output_type = ScalingPerOutputType.GROUP
    restrict_threshold_impl = FloatRestrictValue
    float_to_int_impl = RoundSte  # default rounding type

    @value
    def tracked_parameter_list(upstream_shape):
        return [torch.empty(upstream_shape)]

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


class _GGUFKQuantScalingMixin(__GGUFKQuantScaleZPMixin):
    """Base nested K-quant scale sub-injector (the scale-of-scale)."""
    module = (this << 1).module
    upstream_shape = (this << 1).scaling_shape


class _GGUFKQuantZPMixin(__GGUFKQuantScaleZPMixin):
    """Base nested K-quant zero-point/min sub-injector (the min-of-min)."""
    module = (this << 1).module
    upstream_shape = (this << 1).zero_point_shape


class _GGUFQ6_KScalingSubInjector(_GGUFKQuantScalingMixin, Int8WeightPerChannelFloat):
    """8-bit signed quantizer for the per-sub-block scales (the scale-of-scale)."""
    bit_width = Q6_K_SUB_SCALE_BIT_WIDTH
    group_size = QK_K // Q6_K_GROUP_SIZE
    signed = True
    restrict_scaling_type = RestrictValueType.SIGNED_FP
    scaling_stats_op = StatsOp.SIGNED_MAX


class GGUFQ6_KWeightQuant(_GGUFSignedBaseKQuantMixin, Int8WeightPerChannelFloat):
    """Signed symmetric 6-bit Q6_K super-block quantizer with nested scales."""
    gguf_qtype = gguf.GGMLQuantizationType.Q6_K
    group_size = Q6_K_GROUP_SIZE
    bit_width = 6
    scaling_quant = _GGUFQ6_KScalingSubInjector


class _GGUFQ5_KScalingSubInjector(_GGUFKQuantScalingMixin, Int8WeightPerTensorFloat):
    """6-bit unsigned quantizer for the per-sub-block scales (the scale-of-scale)."""
    bit_width = Q4_K_SUB_SCALE_BIT_WIDTH
    group_size = QK_K // Q4_K_GROUP_SIZE
    signed = False


class _GGUFQ5_KZPSubInjector(_GGUFKQuantZPMixin, Int8WeightPerTensorFloat):
    """6-bit unsigned quantizer for the per-sub-block mins (the min-of-min)."""
    bit_width = Q4_K_SUB_ZP_BIT_WIDTH
    group_size = QK_K // Q4_K_GROUP_SIZE
    signed = False


class GGUFQ5_KWeightQuant(_GGUFShiftedBaseKQuantMixin, ShiftedUint8WeightPerTensorFloat):
    """Asymmetric unsigned 5-bit Q5_K super-block quantizer with nested scales/mins."""
    gguf_qtype = gguf.GGMLQuantizationType.Q5_K
    group_size = Q5_K_GROUP_SIZE
    bit_width = 5
    scaling_quant = _GGUFQ5_KScalingSubInjector
    zp_quant = _GGUFQ5_KZPSubInjector


class GGUFQ4_KWeightQuant(GGUFQ5_KWeightQuant):
    """Asymmetric unsigned 4-bit Q4_K super-block quantizer with nested scales/mins."""
    gguf_qtype = gguf.GGMLQuantizationType.Q4_K
    group_size = Q4_K_GROUP_SIZE
    bit_width = 4


class _GGUFQ3_KScalingSubInjector(_GGUFKQuantScalingMixin, Int8WeightPerChannelFloat):
    """6-bit *signed* quantizer for the per-sub-block scales (the scale-of-scale)."""
    bit_width = Q3_K_SUB_SCALE_BIT_WIDTH
    group_size = QK_K // Q3_K_GROUP_SIZE
    signed = True
    restrict_scaling_type = RestrictValueType.SIGNED_FP
    scaling_stats_op = StatsOp.SIGNED_MAX


class GGUFQ3_KWeightQuant(_GGUFSignedBaseKQuantMixin, Int8WeightPerChannelFloat):
    """Signed symmetric 3-bit Q3_K super-block quantizer with nested scales."""
    gguf_qtype = gguf.GGMLQuantizationType.Q3_K
    group_size = Q3_K_GROUP_SIZE
    bit_width = 3
    scaling_quant = _GGUFQ3_KScalingSubInjector


class _GGUFQ2_KScalingSubInjector(_GGUFKQuantScalingMixin, Int8WeightPerTensorFloat):
    """4-bit unsigned quantizer for the per-sub-block scales (the scale-of-scale)."""
    bit_width = Q2_K_SUB_SCALE_BIT_WIDTH
    group_size = QK_K // Q2_K_GROUP_SIZE
    signed = False
    float_to_int_impl = CeilSte


class _GGUFQ2_KZPSubInjector(_GGUFKQuantZPMixin, Int8WeightPerTensorFloat):
    """4-bit unsigned quantizer for the per-sub-block mins (the min-of-min)."""
    bit_width = Q2_K_SUB_ZP_BIT_WIDTH
    group_size = QK_K // Q2_K_GROUP_SIZE
    signed = False
    float_to_int_impl = FloorSte


class GGUFQ2_KWeightQuant(_GGUFShiftedBaseKQuantMixin, ShiftedUint8WeightPerTensorFloat):
    """Asymmetric unsigned 2-bit Q2_K super-block quantizer with nested scales/mins."""
    gguf_qtype = gguf.GGMLQuantizationType.Q2_K
    group_size = Q2_K_GROUP_SIZE
    bit_width = 2
    scaling_quant = _GGUFQ2_KScalingSubInjector
    zp_quant = _GGUFQ2_KZPSubInjector
