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

from abc import ABCMeta
from dataclasses import dataclass, field, InitVar
from typing import Optional, Union, Tuple, Type

from torch.nn import Module

from brevitas.core.bit_width import *
from brevitas.core.quant import QuantType
from brevitas.core.stats import StatsOp, StatsInputViewShapeImpl
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE


SCALING_MIN_VAL = 2.0 ** (-16)


@dataclass
class QuantConfig:
    bit_width: int = 8
    quant_type: QuantType = QuantType.INT


@dataclass
class ActQuantConfig(QuantConfig):
    layer: InitVar[Module] = None
    min_val: float = 1.0
    max_val: float = -1.0
    signed: bool = True
    narrow_range: bool = False
    quant_type: QuantType = QuantType.FP
    float_to_int_impl_type: FloatToIntImplType = FloatToIntImplType.ROUND
    scaling_impl_type: ScalingImplType = ScalingImplType.PARAMETER
    scaling_override: Optional[Module] = None
    scaling_per_channel: bool = False
    scaling_stats_sigma: float = 3.0
    scaling_stats_op: StatsOp = StatsOp.MEAN_LEARN_SIGMA_STD
    scaling_stats_buffer_momentum: float = 0.1
    scaling_stats_permute_dims: Optional[Tuple] = (1, 0, 2, 3)
    scaling_per_output_channel_reduce_dim = 1
    per_channel_broadcastable_shape: Optional[Tuple[int, ...]] = None
    min_overall_bit_width: Optional[int] = 2
    max_overall_bit_width: Optional[int] = None
    override_pretrained_bit_width = False
    bit_width_impl_override: Union[BitWidthParameter] = None
    bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST
    restrict_bit_width_type: RestrictValueType = RestrictValueType.INT
    restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP
    scaling_min_val: Optional[float] = SCALING_MIN_VAL

    @property
    def scaling_stats_reduce_dim(self):
        if self.scaling_per_channel or self.scaling_stats_op == StatsOp.MAX_AVE:
            return self.scaling_per_output_channel_reduce_dim
        else:
            return None

    @property
    def scaling_stats_input_view_shape_impl(self):
        if self.scaling_per_channel or self.scaling_stats_op.StatsOp.MAX_AVE:
            return StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
        else:
            return StatsInputViewShapeImpl.OVER_TENSOR


@dataclass
class SignedActQuantConfig(ActQuantConfig):
    signed: bool = field(default=True, init=False)


@dataclass
class UnsignedActQuantConfig(ActQuantConfig):
    signed: bool = field(default=False, init=False)
    min_val: float = field(default=0.0, init=False)


@dataclass
class WeightQuantConfig(QuantConfig):
    weight_layer: InitVar[Module] = None
    narrow_range: bool = False
    bit_width_impl_override: BitWidthParameter = None
    bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST
    restrict_bit_width_type: RestrictValueType = RestrictValueType.INT
    min_overall_bit_width: Optional[int] = 2
    max_overall_bit_width: Optional[int] = None
    output_channel_dim: int = 0
    scaling_override: Optional[Module] = None
    scaling_impl_type: ScalingImplType = ScalingImplType.STATS
    scaling_const: Optional[float] = None
    scaling_stats_op: StatsOp = StatsOp.MAX
    scaling_per_output_channel: bool = False
    scaling_per_output_channel_reduce_dim = 1
    scaling_min_val: float = SCALING_MIN_VAL
    ternary_threshold: float = 0.5
    restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP
    scaling_stats_sigma: float = 3.0
    override_pretrained_bit_width: bool = False
    # attributes below are set post init
    scaling_per_output_channel_shape : Tuple[int] = field(default=None, init=False)
    returned_scale_shape: Tuple[int] = field(default=None, init=False)
    scaling_stats_input_concat_dim: int = field(default=None, init=False)

    def __post_init__(self, weight_layer: Module):
        per_channel_brodcast_shape = [1] * len(weight_layer.weight.size())
        per_channel_brodcast_shape[weight_layer.output_channel_dim] = weight_layer.out_channels
        self.scaling_per_output_channel_shape = tuple(per_channel_brodcast_shape)
        self.scaling_stats_input_concat_dim = weight_layer.output_channel_dim
        self.returned_scale_shape = weight_layer.returned_scale_shape

    @property
    def scaling_stats_reduce_dim(self):
        if not self.scaling_per_output_channel or self.scaling_stats_op == StatsOp.MAX_AVE:
            return None
        else:
            return self.scaling_per_output_channel_reduce_dim

    @property
    def scaling_stats_input_view_shape_impl(self):
        if self.scaling_per_output_channel or self.scaling_stats_op == StatsOp.MAX_AVE:
            return StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
        else:
            return StatsInputViewShapeImpl.OVER_TENSOR

    @property
    def scaling_shape(self):
        if self.scaling_per_output_channel:
            return self.scaling_per_output_channel_shape
        else:
            return SCALING_SCALAR_SHAPE


@dataclass
class BiasQuantConfig:
    bias_layer: InitVar[Module] = None
    narrow_range: bool = False
    bit_width: Optional[int] = None
    quant_type: QuantType = QuantType.FP





