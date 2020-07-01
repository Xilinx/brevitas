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
from dataclasses import dataclass, field
from typing import Optional, Union, Tuple

from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.stats import StatsOp
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType, StatsInputViewShapeImpl
from brevitas.quant_tensor import QuantTensor


SCALING_MIN_VAL = 2.0 ** (-16)


@dataclass
class QuantConfig(metaclass=ABCMeta):
    float_to_int_impl_type: FloatToIntImplType = FloatToIntImplType.ROUND
    scaling_impl_type: ScalingImplType = ScalingImplType.PARAMETER
    scaling_override: Optional[Module] = None
    scaling_per_channel: bool = False
    scaling_min_val: Optional[float] = SCALING_MIN_VAL
    scaling_stats_sigma: float
    scaling_stats_op: StatsOp
    scaling_stats_buffer_momentum = 0.1
    scaling_stats_permute_dims = (1, 0, 2, 3)
    per_channel_broadcastable_shape: Optional[Tuple[int, ...]] = None
    min_overall_bit_width: Optional[int] = 2
    max_overall_bit_width: Optional[int] = None
    bit_width_impl_override: Union[BitWidthParameter] = None
    bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST
    restrict_bit_width_type: RestrictValueType = RestrictValueType.INT
    restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP
    override_pretrained_bit_width: bool = False


@dataclass
class QuantActivationConfig(QuantConfig):
    scaling_stats_sigma = 2.0
    scaling_stats_op = StatsOp.MEAN_LEARN_SIGMA_STD


@dataclass
class WeightQuantConfig:
    narrow_range: bool = False
    bit_width_impl_override: Union[BitWidthParameter] = None
    bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST
    restrict_bit_width_type: RestrictValueType = RestrictValueType.INT
    bit_width: int = 32
    min_overall_bit_width: Optional[int] = 2
    max_overall_bit_width: Optional[int] = None
    scaling_override: Optional[Module] = None
    scaling_impl_type: ScalingImplType = ScalingImplType.STATS
    scaling_const: Optional[float] = None
    scaling_stats_op: StatsOp = StatsOp.MAX
    scaling_per_output_channel: bool = False
    scaling_min_val: float = SCALING_MIN_VAL
    ternary_threshold: float = 0.5
    restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP
    scaling_stats_sigma: float = 3.0
    override_pretrained_bit_width: bool = False


@dataclass
class BiasQuantConfig:
    narrow_range: bool = False


class QuantLayer(object):
    __metaclass__ = ABCMeta

    def __init__(self, compute_output_scale, compute_output_bit_width, return_quant_tensor):
        self.compute_output_scale = compute_output_scale
        self.compute_output_bit_width = compute_output_bit_width
        self.return_quant_tensor = return_quant_tensor

    def unpack_input(self, input):
        if isinstance(input, QuantTensor):
            return input
        else:
            return input, None, None

    def pack_output(self,
                    output,
                    output_scale,
                    output_bit_width):
        if self.return_quant_tensor:
            return QuantTensor(tensor=output, scale=output_scale, bit_width=output_bit_width)
        else:
            return output
