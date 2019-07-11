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
from typing import Tuple, Optional, Union, List
from functools import partial

import torch
from torch import nn, Tensor

from brevitas.core.scaling import ScalingImplType, StatsScaling, StatsInputViewShapeImpl, IntScaling
from brevitas.core.scaling import  StandaloneScaling, SCALING_SCALAR_SHAPE
from brevitas.core.stats import StatsOp
from brevitas.core.quant import QuantType, BinaryQuant, TernaryQuant, RescalingIntQuant, PrescaledIntQuant
from brevitas.core.quant import IdentityQuant
from brevitas.core import ZERO_HW_SENTINEL_NAME, ZERO_HW_SENTINEL_VALUE
from brevitas.core.bit_width import BitWidthConst, BitWidthParameter, BitWidthImplType
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType, RestrictValue
from brevitas.core.function_wrapper import TensorClampSte, TensorClamp, RoundSte


__all__ = ['WeightQuantProxy', 'BiasQuantProxy']


OVER_BATCH_OVER_CHANNELS_SHAPE = (1, -1, 1, 1)


class ParameterQuantProxy(nn.Module):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(ParameterQuantProxy, self).__init__()
        self.register_buffer(ZERO_HW_SENTINEL_NAME, torch.tensor(ZERO_HW_SENTINEL_VALUE))

    @property
    def tensor_quant(self):
        return self._tensor_quant

    @tensor_quant.setter
    def tensor_quant(self, tensor_quant):
        self._tensor_quant = tensor_quant

    @tensor_quant.deleter
    def tensor_quant(self):
        del self._tensor_quant


def _weight_quant_init_impl(bit_width: Optional[int],
                            quant_type: QuantType,
                            narrow_range: bool,
                            restrict_scaling_type: RestrictValueType,
                            scaling_stats_op: StatsOp,
                            scaling_impl_type: ScalingImplType,
                            scaling_stats_reduce_dim: Optional[int],
                            scaling_shape: Tuple[int, ...],
                            bit_width_impl_type: Optional[BitWidthImplType],
                            restrict_bit_width_type: Optional[RestrictValueType],
                            min_overall_bit_width: Optional[int],
                            max_overall_bit_width: Optional[int],
                            bit_width_impl_override: Optional[Union[BitWidthConst, BitWidthParameter]],
                            scaling_stats_input_view_shape_impl: StatsInputViewShapeImpl,
                            scaling_stats_input_concat_dim: int,
                            ternary_threshold: Optional[float],
                            scaling_stats_sigma: Optional[float],
                            tracked_parameter_list: List[torch.nn.Parameter],
                            zero_hw_sentinel: torch.Tensor,
                            override_pretrained_bit_width: bool):
    if quant_type == QuantType.FP:
        tensor_quant = IdentityQuant()
    else:
        if scaling_impl_type == ScalingImplType.STATS or scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
            stats_scaling_init = partial(StatsScaling,
                                         stats_op=scaling_stats_op,
                                         restrict_scaling_type=restrict_scaling_type,
                                         tracked_parameter_list=tracked_parameter_list,
                                         stats_input_view_shape_impl=scaling_stats_input_view_shape_impl,
                                         stats_input_concat_dim=scaling_stats_input_concat_dim,
                                         sigma=scaling_stats_sigma,
                                         stats_reduce_dim=scaling_stats_reduce_dim)
            if scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
                stats_scaling = stats_scaling_init(stats_reduce_dim=None,
                                                   stats_output_shape=SCALING_SCALAR_SHAPE)
                scaling_init = stats_scaling(zero_hw_sentinel).detach().item()
                scaling_impl = StandaloneScaling(scaling_init=scaling_init,
                                                 parameter_shape=scaling_shape,
                                                 restrict_scaling_type=restrict_scaling_type,
                                                 is_parameter=True)
            else:
                scaling_impl = stats_scaling_init(stats_reduce_dim=scaling_stats_reduce_dim,
                                                  stats_output_shape=scaling_shape)
        else:
            raise Exception("Scaling type {} not supported for weight quantization"
                            .format(str(scaling_impl_type)))

        if bit_width == 1 and quant_type == QuantType.BINARY:
            tensor_quant = BinaryQuant(scaling_impl=scaling_impl)

        elif bit_width == 2 and quant_type == QuantType.TERNARY:
            tensor_quant = TernaryQuant(scaling_impl=scaling_impl, threshold=ternary_threshold)

        elif bit_width >= 2 and quant_type == QuantType.INT:
            if bit_width_impl_override is None:
                if (bit_width_impl_type is None
                        or bit_width is None
                        or restrict_bit_width_type is None):
                    raise Exception("Bit width is not defined properly")

                if bit_width_impl_type == BitWidthImplType.CONST:
                    tensor_clamp_impl = TensorClampSte()
                    bit_width_impl = BitWidthConst(bit_width, restrict_bit_width_type)
                elif bit_width_impl_type == BitWidthImplType.PARAMETER:
                    tensor_clamp_impl = TensorClamp()
                    bit_width_impl = BitWidthParameter(bit_width_init=bit_width,
                                                       restrict_bit_width_type=restrict_bit_width_type,
                                                       min_overall_bit_width=min_overall_bit_width,
                                                       max_overall_bit_width=max_overall_bit_width,
                                                       override_pretrained=override_pretrained_bit_width)
                else:
                    raise Exception("Bit width type {} not supported for weight quantization."
                                    .format(str(bit_width_impl_type)))
            else:
                tensor_clamp_impl = TensorClamp()
                bit_width_impl = bit_width_impl_override

            float_to_int_impl = RestrictValue(restrict_value_type=RestrictValueType.INT,
                                              float_to_int_impl_type=FloatToIntImplType.ROUND)
            int_scaling_impl = IntScaling(narrow_range,
                                          signed=True,
                                          restrict_scaling_type=restrict_scaling_type)
            tensor_quant = RescalingIntQuant(narrow_range=narrow_range,
                                             signed=True,
                                             scaling_impl=scaling_impl,
                                             int_scaling_impl=int_scaling_impl,
                                             tensor_clamp_impl=tensor_clamp_impl,
                                             msb_clamp_bit_width_impl=bit_width_impl,
                                             float_to_int_impl=float_to_int_impl)
        else:
            raise Exception('Unsupported weight quantization: {} bit width, {} quantization.'
                            .format(bit_width, str(quant_type)))
    return tensor_quant


class WeightQuantProxy(ParameterQuantProxy):

    def __init__(self,
                 bit_width: Optional[int],
                 quant_type: QuantType,
                 narrow_range: bool,
                 restrict_scaling_type: RestrictValueType,
                 scaling_stats_op: StatsOp,
                 scaling_impl_type: ScalingImplType,
                 scaling_stats_reduce_dim: Optional[int],
                 scaling_shape: Tuple[int, ...],
                 bit_width_impl_type: Optional[BitWidthImplType],
                 restrict_bit_width_type: Optional[RestrictValueType],
                 min_overall_bit_width: Optional[int],
                 max_overall_bit_width: Optional[int],
                 tracked_parameter_list_init: torch.nn.Parameter,
                 bit_width_impl_override: Optional[Union[BitWidthConst, BitWidthParameter]],
                 scaling_stats_input_view_shape_impl: StatsInputViewShapeImpl,
                 scaling_stats_input_concat_dim: int,
                 ternary_threshold: Optional[float],
                 scaling_stats_sigma: Optional[float],
                 override_pretrained_bit_width: bool) -> None:
        super(WeightQuantProxy, self).__init__()
        zero_hw_sentinel = getattr(self, ZERO_HW_SENTINEL_NAME)
        self.lazy_tensor_quant_init = partial(_weight_quant_init_impl,
                                              bit_width=bit_width,
                                              quant_type=quant_type,
                                              narrow_range=narrow_range,
                                              restrict_scaling_type=restrict_scaling_type,
                                              scaling_stats_op=scaling_stats_op,
                                              scaling_impl_type=scaling_impl_type,
                                              scaling_stats_reduce_dim=scaling_stats_reduce_dim,
                                              scaling_shape=scaling_shape,
                                              bit_width_impl_type=bit_width_impl_type,
                                              restrict_bit_width_type=restrict_bit_width_type,
                                              min_overall_bit_width=min_overall_bit_width,
                                              max_overall_bit_width=max_overall_bit_width,
                                              bit_width_impl_override=bit_width_impl_override,
                                              scaling_stats_input_view_shape_impl=scaling_stats_input_view_shape_impl,
                                              scaling_stats_input_concat_dim=scaling_stats_input_concat_dim,
                                              ternary_threshold=ternary_threshold,
                                              scaling_stats_sigma=scaling_stats_sigma,
                                              zero_hw_sentinel=zero_hw_sentinel,
                                              override_pretrained_bit_width=override_pretrained_bit_width)
        self._tracked_parameter_list = [tracked_parameter_list_init]
        self.scale_output_shape = OVER_BATCH_OVER_CHANNELS_SHAPE
        self.re_init_tensor_quant()

    def re_init_tensor_quant(self):
        self.tensor_quant = self.lazy_tensor_quant_init(tracked_parameter_list=self._tracked_parameter_list)

    def add_tracked_parameter(self, x: torch.nn.Parameter) -> None:
        self._tracked_parameter_list.append(x)
        if not isinstance(self.tensor_quant, IdentityQuant):
            del self.tensor_quant
            self.re_init_tensor_quant()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        zero_hw_sentinel = getattr(self, ZERO_HW_SENTINEL_NAME)
        out, scale, bit_width = self.tensor_quant(x, zero_hw_sentinel)
        reshaped_scale = scale.view(self.scale_output_shape)
        return out, reshaped_scale, bit_width

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(WeightQuantProxy, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        zero_hw_sentinel_key = prefix + ZERO_HW_SENTINEL_NAME
        if zero_hw_sentinel_key in missing_keys:
            missing_keys.remove(zero_hw_sentinel_key)


class BiasQuantProxy(ParameterQuantProxy):

    def __init__(self,
                 quant_type: QuantType,
                 narrow_range: bool) -> None:
        super(BiasQuantProxy, self).__init__()

        if quant_type == QuantType.FP:
            self.tensor_quant = None
        elif quant_type == QuantType.INT:
            tensor_clamp_impl = TensorClamp()
            float_to_int_impl = RestrictValue(restrict_value_type=RestrictValueType.INT,
                                              float_to_int_impl_type=FloatToIntImplType.ROUND)
            self.tensor_quant = PrescaledIntQuant(narrow_range=narrow_range,
                                                  signed=True,
                                                  tensor_clamp_impl=tensor_clamp_impl,
                                                  float_to_int_impl=float_to_int_impl)
        else:
            raise Exception('Quantization type {} not supported for bias quant.'
                            .format(str(quant_type)))

    def forward(self,
                x: Tensor,
                scale: Tensor,
                bit_width: Tensor) -> Tensor:
        if self.tensor_quant is not None:
            zero_hw_sentinel = getattr(self, ZERO_HW_SENTINEL_NAME)
            reshaped_scale = scale.view(-1)
            out = self.tensor_quant(x, reshaped_scale, bit_width, zero_hw_sentinel)
            return out
        else:
            return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(BiasQuantProxy, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        zero_hw_sentinel_key = prefix + ZERO_HW_SENTINEL_NAME
        if zero_hw_sentinel_key in missing_keys:
            missing_keys.remove(zero_hw_sentinel_key)
