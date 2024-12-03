from dependencies import this
from dependencies import value
import torch

from brevitas.core.quant.int import RescalingIntQuant
from brevitas.core.restrict_val import QuantRestrictValue
from brevitas.core.stats.stats_wrapper import SCALAR_SHAPE
from brevitas.inject.enum import ScalingPerOutputType
import brevitas.nn as qnn
from brevitas.proxy.groupwise_int_parameter_quant import GroupwiseWeightQuantProxyFromInjector
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.shifted_scaled_int import ShiftedUint8WeightPerTensorFloat

ZP_BIT_WIDTH = 6
SCALE_BIT_WIDTH = 5


class QuantScalingInt(Int8WeightPerTensorFloat):
    bit_width = SCALE_BIT_WIDTH
    module = (this << 1).module
    tracked_parameter_list = (this << 1).tracked_parameter_list
    upstream_scaling = (this << 1).scaling_per_output_type
    rescaling_int_quant = RescalingIntQuant

    @value
    def scaling_shape(
            scaling_per_output,
            scaling_per_output_channel_shape,
            expanded_groupwise_shape,
            group_dim,
            upstream_scaling):
        if scaling_per_output == ScalingPerOutputType.TENSOR:
            scaling = SCALAR_SHAPE
        elif scaling_per_output == ScalingPerOutputType.CHANNEL:
            scaling = scaling_per_output_channel_shape
        elif scaling_per_output == ScalingPerOutputType.GROUP:
            # Scaling shape is like expanded_groupwise_shape but has 1 in position group_dim + 1
            assert expanded_groupwise_shape is not None, "Per Group scaling not correctly configured"
            assert group_dim is not None, "Per Group scaling not correctly configured"
            size = list(expanded_groupwise_shape)
            size[group_dim + 1] = 1
            scaling = tuple(size)

        # When quantizing scale of groupwise, there will be one extra dim compared to the normal case
        if upstream_scaling == ScalingPerOutputType.GROUP:
            scaling = list(scaling)
            scaling.insert(-1, 1)
            scaling = tuple(scaling)
        return scaling


from brevitas.core.zero_point import _ScaleShiftQuantZeroPoint


class QuantZPInt(Int8WeightPerTensorFloat):
    module = (this << 1).module
    tracked_parameter_list = (this << 1).tracked_parameter_list
    upstream_scaling = (this << 1).scaling_per_output_type
    rescaling_int_quant = RescalingIntQuant
    bit_width = ZP_BIT_WIDTH
    quantize_zero_point = True
    scaling_per_output_type = ScalingPerOutputType.CHANNEL

    @value
    def scaling_shape(
            scaling_per_output,
            scaling_per_output_channel_shape,
            expanded_groupwise_shape,
            group_dim,
            upstream_scaling):
        if scaling_per_output == ScalingPerOutputType.TENSOR:
            scaling = SCALAR_SHAPE
        elif scaling_per_output == ScalingPerOutputType.CHANNEL:
            scaling = scaling_per_output_channel_shape
        elif scaling_per_output == ScalingPerOutputType.GROUP:
            # Scaling shape is like expanded_groupwise_shape but has 1 in position group_dim + 1
            assert expanded_groupwise_shape is not None, "Per Group scaling not correctly configured"
            assert group_dim is not None, "Per Group scaling not correctly configured"
            size = list(expanded_groupwise_shape)
            size[group_dim + 1] = 1
            scaling = tuple(size)

        # When quantizing scale of groupwise, there will be one extra dim compared to the normal case
        if upstream_scaling == ScalingPerOutputType.GROUP:
            scaling = list(scaling)
            scaling.insert(-1, 1)
            scaling = tuple(scaling)
        return scaling


class QuantScaleQuantZPInt8WeightPerTensorFloat(ShiftedUint8WeightPerTensorFloat):
    proxy_class = GroupwiseWeightQuantProxyFromInjector
    scaling_int_quant = QuantScalingInt
    zp_int = QuantZPInt
    restrict_scaling_impl = QuantRestrictValue
    scaling_per_output_type = ScalingPerOutputType.GROUP
    scale_shift_zero_point_impl = _ScaleShiftQuantZeroPoint
    group_size = 32

    @value
    def restrict_value_float_to_int_impl():
        return this.scaling_int_quant.rescaling_int_quant

    @value
    def zp_int_quant():
        return this.zp_int.rescaling_int_quant


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

    linear = qnn.QuantLinear(64, 768, weight_quant=QuantScaleQuantZPInt8WeightPerTensorFloat)
    for module in linear.modules():
        if isinstance(module, QuantRestrictValue):
            module.register_forward_pre_hook(hook_scale)
    for module in linear.modules():
        if isinstance(module, _ScaleShiftQuantZeroPoint):
            module.register_forward_pre_hook(hook_zp)

    linear(torch.randn(1, 64))
