from dependencies import this
from dependencies import value
import gguf
import torch

from brevitas.core.quant.int import RescalingIntQuant
from brevitas.core.restrict_val import QuantRestrictValue
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.core.zero_point import _ScaleShiftQuantZeroPoint
from brevitas.inject.enum import ScalingPerOutputType
import brevitas.nn as qnn
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.quant.base import ExtendedInjector
from brevitas.quant.base import FloatRestrictValue
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat
from brevitas_examples.llm.gguf_export.quant import ggml_quant

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
    def zp_dequantized_shape(scaling_per_output_type, zero_point_shape):
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
