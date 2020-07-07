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
from typing import Tuple, Optional, List

import math
import torch
from torch import Tensor


from brevitas.core.bit_width import BitWidthConst, BitWidthParameter, BitWidthImplType, IdentityBitWidth
from brevitas.core.function_wrapper import TensorClampSte, TensorClamp
from brevitas.core.quant import IdentityQuant
from brevitas.core.quant import QuantType, BinaryQuant, TernaryQuant, RescalingIntQuant
from brevitas.core.quant import PrescaledRestrictIntQuant, PrescaledRestrictIntQuantWithInputBitWidth
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType, RestrictValue
from brevitas.core.scaling import ScalingImplType, ParameterStatsScaling, IntScaling
from brevitas.core.scaling import ConstScaling, ParameterScaling
from brevitas.function.ops_ste import round_ste
from brevitas import docstrings

from .quant_proxy import QuantProxy
from .config import WeightQuantConfig, BiasQuantConfig

__all__ = ['WeightQuantProxy', 'BiasQuantProxy']


class ParameterQuantProxy(QuantProxy):
    __metaclass__ = ABCMeta

    @property
    def tensor_quant(self):
        return self._tensor_quant

    @tensor_quant.setter
    def tensor_quant(self, tensor_quant):
        self._tensor_quant = tensor_quant

    @tensor_quant.deleter
    def tensor_quant(self):
        del self._tensor_quant


def _scaling_impl_init(
        wqc: WeightQuantConfig,
        tracked_parameter_list: List[torch.nn.Parameter]):

    if wqc.scaling_impl_type != ScalingImplType.OVERRIDE and wqc.scaling_override is not None:
        raise Exception("Overriding scaling requires to set ScalingImplType to OVERRIDE explicitly.")
    if wqc.scaling_impl_type == ScalingImplType.OVERRIDE and wqc.scaling_override is None:
        raise Exception("Overriding scaling requires to pass a scaling impl module.")

    if wqc.scaling_impl_type == ScalingImplType.OVERRIDE and wqc.scaling_override is not None:
        scaling_impl = wqc.scaling_override

    elif wqc.scaling_impl_type == ScalingImplType.STATS \
            or wqc.scaling_impl_type == ScalingImplType.AFFINE_STATS \
            or wqc.scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
        stats_scaling = ParameterStatsScaling(stats_op=wqc.scaling_stats_op,
                                              restrict_scaling_type=wqc.restrict_scaling_type,
                                              tracked_parameter_list=tracked_parameter_list,
                                              stats_input_view_shape_impl=wqc.scaling_stats_input_view_shape_impl,
                                              stats_input_concat_dim=wqc.scaling_stats_input_concat_dim,
                                              sigma=wqc.scaling_stats_sigma,
                                              scaling_min_val=wqc.scaling_min_val,
                                              stats_reduce_dim=wqc.scaling_stats_reduce_dim,
                                              stats_output_shape=wqc.scaling_shape,
                                              affine=wqc.scaling_impl_type == ScalingImplType.AFFINE_STATS)
        if wqc.scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
            if wqc.quant_type == QuantType.BINARY or wqc.quant_type == QuantType.TERNARY:
                raise Exception("Parameter from stats scaling is currently not supported for binary/ternary")
            scaling_init = stats_scaling().detach()
            scaling_impl = ParameterScaling(scaling_init=scaling_init,
                                             parameter_shape=wqc.scaling_shape,
                                             restrict_scaling_type=wqc.restrict_scaling_type,
                                             scaling_min_val=wqc.scaling_min_val)
        else:
            scaling_impl = stats_scaling

    elif wqc.scaling_impl_type == ScalingImplType.CONST or wqc.scaling_impl_type == ScalingImplType.HE:
        if wqc.scaling_impl_type == ScalingImplType.HE:
            scaling_const = 0.0
            for param in tracked_parameter_list:  # takes average of He scaling over parameter list
                two_dim_param = param.view(param.shape[0], -1)
                scaling_const += math.sqrt(2.0 / two_dim_param.shape[1])
            scaling_const /= len(tracked_parameter_list)
        scaling_init = torch.tensor(scaling_const)
        scaling_impl = ConstScaling(scaling_init=scaling_init,
                                         restrict_scaling_type=wqc.restrict_scaling_type,
                                         scaling_min_val=None)
    else:
        raise Exception("Scaling type {} not supported for weight quantization"
                        .format(str(wqc.scaling_impl_type)))
    return scaling_impl


def _bit_width_impl_init(wqc: WeightQuantConfig):
    if wqc.bit_width_impl_override is not None:
        bit_width_impl = wqc.bit_width_impl_override
    elif wqc.bit_width_impl_type == BitWidthImplType.CONST:
            assert wqc.restrict_bit_width_type == RestrictValueType.INT
            bit_width_impl = BitWidthConst(wqc.bit_width)
    elif wqc.bit_width_impl_type == BitWidthImplType.PARAMETER:
            bit_width_impl = BitWidthParameter(
                bit_width_init=wqc.bit_width,
                restrict_bit_width_type=wqc.restrict_bit_width_type,
                min_overall_bit_width=wqc.min_overall_bit_width,
                max_overall_bit_width=wqc.max_overall_bit_width,
                override_pretrained=wqc.override_pretrained_bit_width)
    else:
        raise Exception("Bit width type {} not supported for weight quantization."
                        .format(str(wqc.bit_width_impl_type)))
    return bit_width_impl


def _tensor_quant_init(
        wqc: WeightQuantConfig,
        tracked_parameter_list: List[torch.nn.Parameter]):

    if wqc.quant_type == QuantType.FP:
        tensor_quant = IdentityQuant()
    else:
        scaling_impl = _scaling_impl_init(wqc, tracked_parameter_list)

        if wqc.bit_width == 1 and wqc.quant_type == QuantType.BINARY:
            tensor_quant = BinaryQuant(scaling_impl=scaling_impl)

        elif wqc.bit_width == 2 and wqc.quant_type == QuantType.TERNARY:
            tensor_quant = TernaryQuant(scaling_impl=scaling_impl, threshold=wqc.ternary_threshold)

        elif wqc.bit_width >= 2 and wqc.quant_type == QuantType.INT:
            bit_width_impl = _bit_width_impl_init(wqc)

            if wqc.bit_width_impl_type == BitWidthImplType.PARAMETER or \
                    wqc.bit_width_impl_type == BitWidthImplType.CONST and \
                    wqc.scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
                tensor_clamp_impl = TensorClamp()
            else:
                tensor_clamp_impl = TensorClampSte()

            float_to_int_impl = RestrictValue(
                restrict_value_type=RestrictValueType.INT,
                float_to_int_impl_type=FloatToIntImplType.ROUND,
                min_val=None)
            int_scaling_impl = IntScaling(
                wqc.narrow_range,
                signed=True,
                restrict_scaling_type=wqc.restrict_scaling_type)
            tensor_quant = RescalingIntQuant(
                narrow_range=wqc.narrow_range,
                signed=True,
                scaling_impl=scaling_impl,
                int_scaling_impl=int_scaling_impl,
                tensor_clamp_impl=tensor_clamp_impl,
                msb_clamp_bit_width_impl=bit_width_impl,
                float_to_int_impl=float_to_int_impl)
        else:
            raise Exception('Unsupported weight quantization: {} bit width, {} quantization.'
                            .format(wqc.bit_width, str(wqc.quant_type)))
    return tensor_quant


@docstrings.get_sectionsf('weight_quant_proxy')
class WeightQuantProxy(ParameterQuantProxy):

    def __init__(
            self,
            weight_quant_config: WeightQuantConfig,
            tracked_parameter_list_init: torch.nn.Parameter) -> None:
        super(WeightQuantProxy, self).__init__()
        self.weight_quant_config = weight_quant_config
        self._tracked_parameter_list = [tracked_parameter_list_init]
        self.init_tensor_quant()

    def init_tensor_quant(self):
        self.tensor_quant = _tensor_quant_init(
            self.weight_quant_config, self._tracked_parameter_list)

    def add_tracked_parameter(self, x: torch.nn.Parameter) -> None:
        self._tracked_parameter_list.append(x)
        if not isinstance(self.tensor_quant, IdentityQuant):
            del self.tensor_quant
            self.init_tensor_quant()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, scale, bit_width = self.tensor_quant(x)
        scale = scale.view(self.weight_quant_config.returned_scale_shape)
        return out, scale, bit_width

    def int_weight(self, x: torch.Tensor):
        quant_weight, scale, _ = self.tensor_quant(x)
        quant_weight = quant_weight / scale
        quant_weight = round_ste(quant_weight)
        quant_weight = quant_weight.int()
        return quant_weight

    def _load_from_state_dict(
            self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(WeightQuantProxy, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self.init_tensor_quant()


class BiasQuantProxy(ParameterQuantProxy):

    def __init__(self, bias_quant_config: BiasQuantConfig) -> None:
        super(BiasQuantProxy, self).__init__()
        bqc = bias_quant_config

        if bqc.quant_type == QuantType.FP:
            self.tensor_quant = None
        elif bqc.quant_type == QuantType.INT:
            tensor_clamp_impl = TensorClamp()
            float_to_int_impl = RestrictValue(
                restrict_value_type=RestrictValueType.INT,
                float_to_int_impl_type=FloatToIntImplType.ROUND,
                min_val=None)
            if bqc.bit_width is not None:
                bit_width_impl = BitWidthConst(bqc.bit_width)
                self.tensor_quant = PrescaledRestrictIntQuant(
                    narrow_range=bias_quant_config.narrow_range,
                    signed=True,
                    tensor_clamp_impl=tensor_clamp_impl,
                    msb_clamp_bit_width_impl=bit_width_impl,
                    float_to_int_impl=float_to_int_impl)
                self.requires_input_bit_width = False
            else:
                msb_clamp_bit_width_impl = IdentityBitWidth()
                self.tensor_quant = PrescaledRestrictIntQuantWithInputBitWidth(
                    narrow_range=bias_quant_config.narrow_range,
                    signed=True,
                    tensor_clamp_impl=tensor_clamp_impl,
                    msb_clamp_bit_width_impl=msb_clamp_bit_width_impl,
                    float_to_int_impl=float_to_int_impl)
                self.requires_input_bit_width = True
        else:
            raise Exception('Quantization type {} not supported for bias quant.'
                            .format(str(bqc.quant_type)))

    def forward(
            self,
            x: Tensor,
            input_scale: Tensor,
            input_bit_width: Optional[Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.tensor_quant is not None:

            if input_scale is None:
                raise Exception("Input scale can't be None when quantizing bias")
            input_scale = input_scale.view(-1)

            if self.requires_input_bit_width:  # bit width is defined outside
                if input_bit_width is None:
                    raise Exception("Input bit width can't be None when quantizing bias without a predefined bit width")
                out, output_scale, bias_bit_width = self.tensor_quant(x, input_scale, input_bit_width)
            else:
                out, output_scale, bias_bit_width = self.tensor_quant(x, input_scale)
            output_scale = output_scale.view(self.scale_output_shape)
            return out, output_scale, bias_bit_width
        else:
            return x, input_scale, input_bit_width


