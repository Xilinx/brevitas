# Copyright (c) 2018-     Xilinx, Inc              (Alessandro Pappalardo)
# Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
# Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
# Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
# Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
# Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
# Copyright (c) 2011-2013 NYU                      (Clement Farabet)
# Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
# Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
# Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.

# 3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
#    NEC Laboratories America and IDIAP Research Institute nor the names
#    of its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from enum import auto
from typing import Union, Optional, Tuple
import re

import math
import torch
import docrep
from torch.nn import Conv2d, Module
from torch.nn import functional as F
from torch.nn.functional import conv2d
from torch.nn.parameter import Parameter

from brevitas.core.bit_width import BitWidthParameter, BitWidthConst, BitWidthImplType
from brevitas.core.quant import QuantType, IdentityQuant
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE
from brevitas.core.stats import StatsInputViewShapeImpl, StatsOp
from brevitas.function.ops import ceil_ste, max_uint
from brevitas.proxy.parameter_quant import WeightQuantProxy, BiasQuantProxy, WeightReg
from brevitas.utils.python_utils import AutoName
from brevitas.nn.quant_bn import mul_add_from_bn, QuantBatchNorm2d
from brevitas.nn.quant_layer import QuantLayer, SCALING_MIN_VAL
from brevitas.config import docstrings

__all__ = ['QuantConv2d']


class PaddingType(AutoName):
    STANDARD = auto()
    SAME = auto()


@docstrings.dedent
class QuantConv2d(QuantLayer, Conv2d):
    """

        Parameters
        ----------

        %(weight_quant_proxy.parameters_with_prefix)s
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 padding_type: PaddingType = PaddingType.STANDARD,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 bias_quant_type: QuantType = QuantType.FP,
                 bias_narrow_range: bool = False,
                 bias_bit_width: int = None,
                 weight_quant_override: WeightQuantProxy = None,
                 weight_quant_type: QuantType = QuantType.FP,
                 weight_narrow_range: bool = False,
                 weight_scaling_override: Optional[Module] = None,
                 weight_bit_width_impl_override: Union[BitWidthParameter, BitWidthConst] = None,
                 weight_bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST,
                 weight_restrict_bit_width_type: RestrictValueType = RestrictValueType.INT,
                 weight_bit_width: int = 32,
                 weight_min_overall_bit_width: Optional[int] = 2,
                 weight_max_overall_bit_width: Optional[int] = None,
                 weight_scaling_impl_type: ScalingImplType = ScalingImplType.STATS,
                 weight_scaling_const: Optional[float] = None,
                 weight_scaling_stats_op: StatsOp = StatsOp.MAX,
                 weight_scaling_per_output_channel: bool = False,
                 weight_ternary_threshold: float = 0.5,
                 weight_restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP,
                 weight_scaling_stats_sigma: float = 3.0,
                 weight_scaling_min_val: float = SCALING_MIN_VAL,
                 weight_override_pretrained_bit_width: bool = False,
                 compute_output_scale: bool = False,
                 compute_output_bit_width: bool = False,
                 return_quant_tensor: bool = False) -> None:
        QuantLayer.__init__(self,
                            compute_output_scale=compute_output_scale,
                            compute_output_bit_width=compute_output_bit_width,
                            return_quant_tensor=return_quant_tensor)
        Conv2d.__init__(self,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        dilation=dilation,
                        groups=groups,
                        bias=bias)
        if weight_quant_type == QuantType.FP and compute_output_bit_width:
            raise Exception("Computing output bit width requires enabling quantization")
        if bias_quant_type != QuantType.FP and not (compute_output_scale and compute_output_bit_width):
            raise Exception("Quantizing bias requires to compute output scale and output bit width")

        self.per_elem_ops = 2 * self.kernel_size[0] * self.kernel_size[1] * (in_channels // groups)
        self.padding_type = padding_type
        self.weight_reg = WeightReg()

        if weight_quant_override is not None:
            self.weight_quant = weight_quant_override
            self.weight_quant.add_tracked_parameter(self.weight)
        else:
            weight_scaling_stats_input_concat_dim = 1
            if weight_scaling_per_output_channel:
                weight_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
                weight_scaling_shape = self.per_output_channel_broadcastable_shape
                weight_scaling_stats_reduce_dim = 1
            else:
                weight_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_TENSOR
                weight_scaling_shape = SCALING_SCALAR_SHAPE
                weight_scaling_stats_reduce_dim = None

            if weight_scaling_stats_op == StatsOp.MAX_AVE:
                weight_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
                weight_scaling_stats_reduce_dim = 1

            self.weight_quant = WeightQuantProxy(bit_width=weight_bit_width,
                                                 quant_type=weight_quant_type,
                                                 narrow_range=weight_narrow_range,
                                                 scaling_override=weight_scaling_override,
                                                 restrict_scaling_type=weight_restrict_scaling_type,
                                                 scaling_const=weight_scaling_const,
                                                 scaling_stats_op=weight_scaling_stats_op,
                                                 scaling_impl_type=weight_scaling_impl_type,
                                                 scaling_stats_reduce_dim=weight_scaling_stats_reduce_dim,
                                                 scaling_shape=weight_scaling_shape,
                                                 bit_width_impl_type=weight_bit_width_impl_type,
                                                 bit_width_impl_override=weight_bit_width_impl_override,
                                                 restrict_bit_width_type=weight_restrict_bit_width_type,
                                                 min_overall_bit_width=weight_min_overall_bit_width,
                                                 max_overall_bit_width=weight_max_overall_bit_width,
                                                 tracked_parameter_list_init=self.weight,
                                                 ternary_threshold=weight_ternary_threshold,
                                                 scaling_stats_input_view_shape_impl=weight_stats_input_view_shape_impl,
                                                 scaling_stats_input_concat_dim=weight_scaling_stats_input_concat_dim,
                                                 scaling_stats_sigma=weight_scaling_stats_sigma,
                                                 scaling_min_val=weight_scaling_min_val,
                                                 override_pretrained_bit_width=weight_override_pretrained_bit_width)
        self.bias_quant = BiasQuantProxy(quant_type=bias_quant_type,
                                         bit_width=bias_bit_width,
                                         narrow_range=bias_narrow_range)

    @property
    def per_output_channel_broadcastable_shape(self):
        if self.transposed:
            raise Exception("Transposed filters are not supported.")
        else:
            output_dim = 0
        per_channel_size = [1] * len(self.weight.size())
        per_channel_size[output_dim] = self.out_channels
        per_channel_size = tuple(per_channel_size)
        return per_channel_size

    @property
    def int_weight(self):
        if isinstance(self.weight_quant.tensor_quant, IdentityQuant):
            raise Exception("Can't export int weight without quantization enabled")
        return self.weight_quant.int_weight(self.weight)

    @property
    def quant_weight_scale(self):
        """

        Returns scale factor of the quantized weights with scalar () shape or (self.out_channels, 1, 1, 1)
        shape depending on whether scaling is per layer or per-channel.
        -------

        """
        if isinstance(self.weight_quant.tensor_quant, IdentityQuant):
            raise Exception("Can't generate scaling factor without quantization enabled")
        zero_hw_sentinel = self.weight_quant.zero_hw_sentinel
        _, scale, _ = self.weight_quant.tensor_quant(self.weight, zero_hw_sentinel)
        return scale

    def forward(self, input):
        output_scale = None
        output_bit_width = None
        bias_bit_width = None

        input, input_scale, input_bit_width = self.unpack_input(input)
        quant_weight, quant_weight_scale, quant_weight_bit_width = self.weight_quant(self.weight)
        quant_weight = self.weight_reg(quant_weight)

        if self.compute_output_bit_width:
            output_bit_width = self.max_output_bit_width(input_bit_width, quant_weight_bit_width)
        if self.compute_output_scale:
            output_scale = input_scale * quant_weight_scale

        if self.bias is not None:
            quant_bias, _, bias_bit_width = self.bias_quant(self.bias, output_scale, output_bit_width)
            output = self.conv2d(input, quant_weight, quant_bias)
        else:
            output = self.conv2d(input, quant_weight, None)

        if self.compute_output_bit_width and bias_bit_width is not None:
            output_bit_width = torch.where(bias_bit_width > output_bit_width, bias_bit_width, output_bit_width)

        return self.pack_output(output, output_scale, output_bit_width)

    def conv2d(self, x, weight, bias):
        if self.padding_type == PaddingType.SAME:
            out = self.conv2d_same_padding(x, weight, bias)
        else:
            out = conv2d(x, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def conv2d_same_padding(self, x, weight, bias):
        ih, iw = x.size()[-2:]
        kh, kw = weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        out = F.conv2d(x, weight, bias, self.stride, 0, self.dilation, self.groups)
        return out

    def merge_bn_in(self, bn, affine_only, sign_only):
        if sign_only and not isinstance(bn, QuantBatchNorm2d):
            raise Exception("Sign-only supported only with QuantBatchNorm2d")
        if affine_only and not bn.affine:
            raise Exception("Affine-only merging requires BN to have affine scaling enabled.")
        if sign_only:
            self.weight.data *= bn.weight_sign.view(self.per_output_channel_broadcastable_shape)
        else:
            mul_factor, add_factor = mul_add_from_bn(bn_mean=bn.running_mean,
                                                     bn_var=bn.running_var,
                                                     bn_eps=bn.eps,
                                                     bn_weight=bn.weight.data.clone(),
                                                     bn_bias=bn.bias.data.clone(),
                                                     affine_only=affine_only)
            self.weight.data *= mul_factor.view(self.per_output_channel_broadcastable_shape)
            if self.bias is not None:
                self.bias.data += add_factor
            else:
                self.bias = Parameter(add_factor)

    def max_output_bit_width(self, input_bit_width, weight_bit_width):
        max_uint_input = max_uint(bit_width=input_bit_width, narrow_range=False)
        max_kernel_val = self.weight_quant.tensor_quant.int_quant.max_uint(weight_bit_width)
        group_size = self.out_channels // self.groups
        max_uint_output = max_uint_input * max_kernel_val * self.kernel_size[0] * self.kernel_size[1] * group_size
        max_output_bit_width = ceil_ste(torch.log2(max_uint_output))
        return max_output_bit_width
