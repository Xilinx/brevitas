# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dependencies import this
from dependencies import value
import pytest
import torch

from brevitas.core.quant.int import RescalingIntQuant
from brevitas.core.restrict_val import QuantRestrictValue
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.core.zero_point import _ScaleShiftQuantZeroPoint
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.nn import QuantLinear
import brevitas.nn as qnn
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.quant.base import ExtendedInjector
from brevitas.quant.base import FloatRestrictValue
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerChannelFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat

ZP_BIT_WIDTH = 6
SCALE_BIT_WIDTH = 6


class ShapeMixin(ExtendedInjector):

    @value
    def scaling_shape(
            scaling_per_output_type,
            scaling_per_output_channel_shape,
            expanded_groupwise_shape,
            group_dim):
        if scaling_per_output_type == ScalingPerOutputType.TENSOR:
            scaling = SCALAR_SHAPE
        elif scaling_per_output_type == ScalingPerOutputType.CHANNEL:
            scaling = scaling_per_output_channel_shape
        elif scaling_per_output_type == ScalingPerOutputType.GROUP:
            # Scaling shape is like expanded_groupwise_shape but has 1 in position group_dim + 1
            assert expanded_groupwise_shape is not None, "Per Group scaling not correctly configured"
            assert group_dim is not None, "Per Group scaling not correctly configured"
            size = list(expanded_groupwise_shape)
            size[group_dim + 1] = 1
            return tuple(size)

        return scaling


class QuantScalingInt(Int8WeightPerTensorFloat, ShapeMixin):
    bit_width = SCALE_BIT_WIDTH
    module = (this << 1).module

    rescaling_int_quant = RescalingIntQuant
    group_size = 8
    scaling_per_output_type = ScalingPerOutputType.GROUP
    upstream_shape = (this << 1).scaling_shape
    signed = False

    @value
    def tracked_parameter_list(upstream_shape):
        return [torch.empty(upstream_shape)]


class QuantZPInt(Int8WeightPerTensorFloat, ShapeMixin):
    module = (this << 1).module

    rescaling_int_quant = RescalingIntQuant
    restrict_threshold_impl = FloatRestrictValue
    bit_width = ZP_BIT_WIDTH
    scaling_per_output_type = ScalingPerOutputType.GROUP
    group_size = 8
    upstream_shape = (this << 1).zero_point_shape
    signed = False

    @value
    def tracked_parameter_list(upstream_shape):
        return [torch.empty(upstream_shape)]


class QuantScaleQuantZPInt8WeightPerTensorFloat(ShiftedUint8WeightPerTensorFloat):
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_quant = QuantScalingInt
    zp_quant = QuantZPInt
    restrict_scaling_impl = QuantRestrictValue
    scaling_per_output_type = ScalingPerOutputType.GROUP
    restrict_threshold_impl = FloatRestrictValue
    scale_shift_zero_point_impl = _ScaleShiftQuantZeroPoint
    group_size = 32
    bit_width = 4

    @value
    def restrict_value_float_to_int_impl():
        return this.scaling_quant.rescaling_int_quant

    @value
    def zp_int_quant():
        return this.zp_quant.rescaling_int_quant

    @value
    def scale_dequantized_shape(scaling_per_output_type, scaling_shape):
        if scaling_per_output_type == ScalingPerOutputType.TENSOR or scaling_per_output_type == ScalingPerOutputType.CHANNEL:
            return None
        elif scaling_per_output_type == ScalingPerOutputType.GROUP:
            return scaling_shape

    @value
    def zero_point_dequantized_shape(scaling_per_output_type, zero_point_shape):
        if scaling_per_output_type == ScalingPerOutputType.TENSOR or scaling_per_output_type == ScalingPerOutputType.CHANNEL:
            return None
        elif scaling_per_output_type == ScalingPerOutputType.GROUP:
            return zero_point_shape


def test_quant_scale():

    def hook_scale(module, inp):
        inp = inp[0]
        quant_scale, scale, zp, bit_width = module.float_to_int_impl(inp)
        assert bit_width == SCALE_BIT_WIDTH
        assert torch.allclose(quant_scale / scale, torch.round(quant_scale / scale))

    def hook_zp(module, inp):
        inp = inp[0]
        quant_scale, scale, zp, bit_width = module.zp_int_quant(inp)
        assert bit_width == ZP_BIT_WIDTH
        assert torch.allclose(quant_scale / scale, torch.round(quant_scale / scale))

    linear = qnn.QuantLinear(512, 768, weight_quant=QuantScaleQuantZPInt8WeightPerTensorFloat)
    for module in linear.modules():
        if isinstance(module, QuantRestrictValue):
            module.register_forward_pre_hook(hook_scale)
    for module in linear.modules():
        if isinstance(module, _ScaleShiftQuantZeroPoint):
            module.register_forward_pre_hook(hook_zp)

    linear(torch.randn(1, 512))


class Uint8ScaledMinMaxPerChannelFloat(ShiftedUint8WeightPerChannelFloat):
    """
    Integer weight quantizer with per-channel scaling factors and zero-points that
    map to a scaled min-max grid. The zero-point is calculated via NegativeMinOrZero,
    the scale factor is calculated via AbsMinMax and normalized via 2^{bit_width} - 1
    """
    quantize_zero_point = False
    scaling_affine_rescaling_init = 0.9
    zero_point_affine_rescaling_init = this.scaling_affine_rescaling_init


class Uint8ScaledMinMaxPerTensorFloat(Uint8ScaledMinMaxPerChannelFloat,
                                      ShiftedUint8WeightPerTensorFloat):
    pass


scaled_weight_quant_map = {
    "per_channel": Uint8ScaledMinMaxPerChannelFloat, "per_tensor": Uint8ScaledMinMaxPerTensorFloat}


@pytest.mark.parametrize(
    "scaled_weight_quant", scaled_weight_quant_map.values(), ids=scaled_weight_quant_map.keys())
@pytest.mark.parametrize("quantize_zero_point", [True, False])
@pytest.mark.parametrize("affine_rescaling_init", [0.9, 0.5])
def test_affine_rescaling_per_channel(
        scaled_weight_quant, quantize_zero_point, affine_rescaling_init, request):

    test_id = request.node.callspec.id

    torch.manual_seed(0)

    scaled_weight_quant = scaled_weight_quant.let(quantize_zero_point=quantize_zero_point)
    scaled_weight_quant = scaled_weight_quant.let(
        scaling_affine_rescaling_init=affine_rescaling_init)
    scaled_weight_quant = scaled_weight_quant.let(
        zero_point_affine_rescaling_init=affine_rescaling_init)

    quant_layer = QuantLinear(256, 32, weight_quant=scaled_weight_quant)
    quant_layer.weight.data = torch.randn(32, 256)
    quant_tensor = quant_layer.quant_weight()

    w = quant_layer.weight.detach()
    q = quant_tensor.value.detach()

    if not quantize_zero_point:
        z = (quant_tensor.zero_point * quant_tensor.scale).squeeze()
        m = -w.min(dim=1).values  # NegativeMinOrZero
        assert torch.allclose((z / m) - affine_rescaling_init, torch.zeros_like(m), atol=1e-3)

    if "per_channel" in test_id:
        q_range = q.max(dim=1).values - q.min(dim=1).values  # range of quant values
        w_range = w.max(dim=1).values - w.min(dim=1).values  # range of float values
    else:
        q_range = q.max() - q.min()  # range of quant values
        w_range = w.max() - w.min()  # range of float values
    assert torch.allclose((q_range / w_range) - affine_rescaling_init,
                          torch.zeros_like(q_range),
                          atol=1e-3)
