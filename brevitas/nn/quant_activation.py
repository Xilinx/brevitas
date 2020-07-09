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
from typing import Optional, Union, Tuple

from torch import nn
from torch.nn import Module

from brevitas.core.bit_width import BitWidthParameter, BitWidthImplType
from brevitas.core.function_wrapper import Identity, ConstScalarClamp
from brevitas.core.quant import QuantType, IdentityQuant
from brevitas.core.stats import StatsOp
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType
from brevitas.core.scaling import ScalingImplType
from brevitas.proxy.runtime_quant import ActivationQuantProxy

from .quant_layer import QuantLayer


class QuantActivation(QuantLayer, Module):
    __metaclass__ = ABCMeta

    def __init__(self, return_quant_tensor):
        QuantLayer.__init__(self, return_quant_tensor=return_quant_tensor)
        Module.__init__(self)

    @property
    def act_quant_proxy(self):
        return self._act_quant_proxy

    @act_quant_proxy.setter
    def act_quant_proxy(self, act_quant_proxy):
        self._act_quant_proxy = act_quant_proxy

    def quant_act_scale(self):
        if isinstance(self.act_quant_proxy.fused_activation_quant_proxy.tensor_quant, IdentityQuant):
            raise Exception("Can't generate scaling factor without quantization enabled")
        scaling_impl = self.act_quant_proxy.fused_activation_quant_proxy.tensor_quant.scaling_impl
        current_status = scaling_impl.training
        scaling_impl.eval()
        _, out, _ = self.act_quant_proxy(self.act_quant_proxy._zero_hw_sentinel())
        scaling_impl.train(current_status)
        return out

    def forward(self, input):
        tensor, _, _ = self.unpack_input(input)
        output, output_scale, output_bit_width = self.act_quant_proxy(tensor)
        return self.pack_output(output, output_scale, output_bit_width)


class QuantReLU(QuantActivation):

    def __init__(self,
                 max_val: float,
                 output_quant_config,
                 return_quant_tensor: bool = False):
        super(QuantReLU, self).__init__(return_quant_tensor=return_quant_tensor)
        activation_impl = nn.ReLU()
        self.act_quant_proxy = ActivationQuantProxy(activation_impl=activation_impl,
                                                    bit_width=bit_width,
                                                    signed=False,
                                                    narrow_range=False,
                                                    scaling_override=scaling_override,
                                                    min_val=0.0,
                                                    max_val=max_val,
                                                    quant_type=quant_type)


class QuantSigmoid(QuantActivation):

    def __init__(self,
                 bit_width: int,
                 narrow_range: bool = False,
                 quant_type: QuantType = QuantType.FP,
                 float_to_int_impl_type: FloatToIntImplType = FloatToIntImplType.ROUND,
                 min_overall_bit_width: Optional[int] = 2,
                 max_overall_bit_width: Optional[int] = None,
                 bit_width_impl_override: Union[BitWidthParameter] = None,
                 bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST,
                 restrict_bit_width_type: RestrictValueType = RestrictValueType.INT,
                 restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP,
                # scaling_min_val: Optional[float],
                 override_pretrained_bit_width: bool = False,
                 return_quant_tensor = False):
        super(QuantSigmoid, self).__init__(return_quant_tensor=return_quant_tensor)
        activation_impl = nn.Sigmoid()
        self.act_quant_proxy = ActivationQuantProxy(activation_impl=activation_impl,
                                                    bit_width=bit_width,
                                                    signed=False,
                                                    narrow_range=narrow_range,
                                                    scaling_override=None,
                                                    min_val=0.0,
                                                    max_val=1.0,
                                                    quant_type=quant_type,
                                                    float_to_int_impl_type=float_to_int_impl_type,
                                                    scaling_impl_type=ScalingImplType.CONST,
                                                    scaling_per_channel=False,
                                                    scaling_min_val=scaling_min_val,
                                                    per_channel_broadcastable_shape=None,
                                                    min_overall_bit_width=min_overall_bit_width,
                                                    max_overall_bit_width=max_overall_bit_width,
                                                    bit_width_impl_override=bit_width_impl_override,
                                                    bit_width_impl_type=bit_width_impl_type,
                                                    restrict_bit_width_type=restrict_bit_width_type,
                                                    restrict_scaling_type=restrict_scaling_type,
                                                    override_pretrained_bit_width=override_pretrained_bit_width,
                                                    scaling_stats_sigma=None,
                                                    scaling_stats_op=None,
                                                    scaling_stats_buffer_momentum=None,
                                                    scaling_stats_permute_dims=None)


class QuantTanh(QuantActivation):

    def __init__(self,
                 bit_width: int,
                 narrow_range: bool = False,
                 quant_type: QuantType = QuantType.FP,
                 float_to_int_impl_type: FloatToIntImplType = FloatToIntImplType.ROUND,
                 min_overall_bit_width: Optional[int] = 2,
                 max_overall_bit_width: Optional[int] = None,
                 bit_width_impl_override: Union[BitWidthParameter] = None,
                 bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST,
                 restrict_bit_width_type: RestrictValueType = RestrictValueType.INT,
                 restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP,
#                 scaling_min_val: Optional[float] = SCALING_MIN_VAL,
                 override_pretrained_bit_width: bool = False,
                 return_quant_tensor: bool = False):
        super(QuantTanh, self).__init__(return_quant_tensor=return_quant_tensor)
        activation_impl = nn.Tanh()
        self.act_quant_proxy = ActivationQuantProxy(activation_impl=activation_impl,
                                                    bit_width=bit_width,
                                                    signed=True,
                                                    narrow_range=narrow_range,
                                                    scaling_override=None,
                                                    min_val=-1.0,
                                                    max_val=1.0,
                                                    quant_type=quant_type,
                                                    float_to_int_impl_type=float_to_int_impl_type,
                                                    scaling_impl_type=ScalingImplType.CONST,
                                                    scaling_per_channel=False,
                                                    scaling_min_val=scaling_min_val,
                                                    per_channel_broadcastable_shape=None,
                                                    min_overall_bit_width=min_overall_bit_width,
                                                    max_overall_bit_width=max_overall_bit_width,
                                                    bit_width_impl_override=bit_width_impl_override,
                                                    bit_width_impl_type=bit_width_impl_type,
                                                    restrict_bit_width_type=restrict_bit_width_type,
                                                    restrict_scaling_type=restrict_scaling_type,
                                                    override_pretrained_bit_width=override_pretrained_bit_width,
                                                    scaling_stats_sigma=None,
                                                    scaling_stats_op=None,
                                                    scaling_stats_buffer_momentum=None,
                                                    scaling_stats_permute_dims=None)


class QuantHardTanh(QuantActivation):

    def __init__(self,
                 bit_width: int,
                 min_val: float = -1.0,
                 max_val: float = 1.0,
                 narrow_range: bool = False,
                 quant_type: QuantType = QuantType.FP,
                 float_to_int_impl_type: FloatToIntImplType = FloatToIntImplType.ROUND,
                 scaling_impl_type: ScalingImplType = ScalingImplType.PARAMETER,
                 scaling_override: Optional[Module] = None,
                 scaling_per_channel: bool = False,
                 scaling_stats_sigma: float = 3.0,
                 scaling_stats_op: StatsOp = StatsOp.MEAN_LEARN_SIGMA_STD,
                 scaling_stats_buffer_momentum: float = 0.1,
                 scaling_stats_permute_dims: Tuple = (1, 0, 2, 3),
                 per_channel_broadcastable_shape: Optional[Tuple[int, ...]] = None,
                 min_overall_bit_width: Optional[int] = 2,
                 max_overall_bit_width: Optional[int] = None,
                 bit_width_impl_override: Union[BitWidthParameter] = None,
                 bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST,
                 restrict_bit_width_type: RestrictValueType = RestrictValueType.INT,
                 restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP,
             #    scaling_min_val: Optional[float] = SCALING_MIN_VAL,
                 override_pretrained_bit_width: bool = False,
                 return_quant_tensor: bool = False):
        super(QuantHardTanh, self).__init__(return_quant_tensor=return_quant_tensor)
        if quant_type == QuantType.FP:
            activation_impl = ConstScalarClamp(min_val=min_val, max_val=max_val)
        else:
            activation_impl = Identity()
        self.act_quant_proxy = ActivationQuantProxy(activation_impl=activation_impl,
                                                    bit_width=bit_width,
                                                    signed=True,
                                                    narrow_range=narrow_range,
                                                    scaling_override=scaling_override,
                                                    min_val=min_val,
                                                    max_val=max_val,
                                                    quant_type=quant_type,
                                                    float_to_int_impl_type=float_to_int_impl_type,
                                                    scaling_impl_type=scaling_impl_type,
                                                    scaling_per_channel=scaling_per_channel,
                                                    scaling_min_val=scaling_min_val,
                                                    per_channel_broadcastable_shape=per_channel_broadcastable_shape,
                                                    min_overall_bit_width=min_overall_bit_width,
                                                    max_overall_bit_width=max_overall_bit_width,
                                                    bit_width_impl_override=bit_width_impl_override,
                                                    bit_width_impl_type=bit_width_impl_type,
                                                    restrict_bit_width_type=restrict_bit_width_type,
                                                    restrict_scaling_type=restrict_scaling_type,
                                                    override_pretrained_bit_width=override_pretrained_bit_width,
                                                    scaling_stats_sigma=scaling_stats_sigma,
                                                    scaling_stats_op=scaling_stats_op,
                                                    scaling_stats_buffer_momentum=scaling_stats_buffer_momentum,
                                                    scaling_stats_permute_dims=scaling_stats_permute_dims)


class QuantIdentity(QuantActivation):

    def __init__(self,
                 bit_width: int,
                 min_val: float = -1.0,
                 max_val: float = 1.0,
                 narrow_range: bool = False,
                 quant_type: QuantType = QuantType.FP,
                 float_to_int_impl_type: FloatToIntImplType = FloatToIntImplType.ROUND,
                 scaling_impl_type: ScalingImplType = ScalingImplType.PARAMETER,
                 scaling_override: Optional[Module] = None,
                 scaling_per_channel: bool = False,
                 scaling_stats_sigma: float = 3.0,
                 scaling_stats_op: StatsOp = StatsOp.MEAN_LEARN_SIGMA_STD,
                 scaling_stats_buffer_momentum: float = 0.1,
                 scaling_stats_permute_dims: Tuple = (1, 0, 2, 3),
                 per_channel_broadcastable_shape: Optional[Tuple[int, ...]] = None,
                 min_overall_bit_width: Optional[int] = 2,
                 max_overall_bit_width: Optional[int] = None,
                 bit_width_impl_override: Union[BitWidthParameter] = None,
                 bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST,
                 restrict_bit_width_type: RestrictValueType = RestrictValueType.INT,
                 restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP,
             #    scaling_min_val: Optional[float] = SCALING_MIN_VAL,
                 override_pretrained_bit_width: bool = False,
                 return_quant_tensor: bool = False):
        super(QuantIdentity, self).__init__(return_quant_tensor=return_quant_tensor)
        activation_impl = Identity()
        self.act_quant_proxy = ActivationQuantProxy(activation_impl=activation_impl,
                                                    bit_width=bit_width,
                                                    signed=True,
                                                    narrow_range=narrow_range,
                                                    scaling_override=scaling_override,
                                                    min_val=min_val,
                                                    max_val=max_val,
                                                    quant_type=quant_type,
                                                    float_to_int_impl_type=float_to_int_impl_type,
                                                    scaling_impl_type=scaling_impl_type,
                                                    scaling_per_channel=scaling_per_channel,
                                                    scaling_min_val=scaling_min_val,
                                                    per_channel_broadcastable_shape=per_channel_broadcastable_shape,
                                                    min_overall_bit_width=min_overall_bit_width,
                                                    max_overall_bit_width=max_overall_bit_width,
                                                    bit_width_impl_override=bit_width_impl_override,
                                                    bit_width_impl_type=bit_width_impl_type,
                                                    restrict_bit_width_type=restrict_bit_width_type,
                                                    restrict_scaling_type=restrict_scaling_type,
                                                    override_pretrained_bit_width=override_pretrained_bit_width,
                                                    scaling_stats_sigma=scaling_stats_sigma,
                                                    scaling_stats_op=scaling_stats_op,
                                                    scaling_stats_buffer_momentum=scaling_stats_buffer_momentum,
                                                    scaling_stats_permute_dims=scaling_stats_permute_dims)
