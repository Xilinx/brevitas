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
from functools import partial
from typing import Tuple, Optional, Union, List
import re

import math
import torch
from torch import nn, Tensor


from brevitas.core import ZERO_HW_SENTINEL_NAME
from brevitas.core.bit_width import BitWidthConst, BitWidthParameter, BitWidthImplType
from brevitas.core.function_wrapper import TensorClampSte, TensorClamp
from brevitas.core.quant import IdentityQuant
from brevitas.core.quant import QuantType, BinaryQuant, TernaryQuant, RescalingIntQuant, PrescaledIntQuant
from brevitas.core.quant import PrescaledRestrictIntQuant
from brevitas.core.restrict_val import RestrictValueType, FloatToIntImplType, RestrictValue
from brevitas.core.scaling import ScalingImplType, ParameterStatsScaling, StatsInputViewShapeImpl, IntScaling
from brevitas.core.scaling import StandaloneScaling, SCALING_SCALAR_SHAPE
from brevitas.function.ops import round_ste
from brevitas.core.stats import StatsOp
from brevitas import config
from brevitas.config import docstrings

from .quant_proxy import QuantProxy

__all__ = ['WeightQuantProxy', 'BiasQuantProxy']


OVER_BATCH_OVER_CHANNELS_SHAPE = (1, -1, 1, 1)


class WeightReg(nn.Module):

    def __init__(self):
        super(WeightReg, self).__init__()
        pass

    def forward(self, weight):
        return weight + 0


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


def _weight_quant_init_impl(bit_width: Optional[int],
                            quant_type: QuantType,
                            narrow_range: bool,
                            scaling_override: Optional[nn.Module],
                            restrict_scaling_type: RestrictValueType,
                            scaling_const: float,
                            scaling_stats_op: StatsOp,
                            scaling_impl_type: ScalingImplType,
                            scaling_stats_reduce_dim: Optional[int],
                            scaling_shape: Tuple[int, ...],
                            scaling_min_val: Optional[float],
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
        if scaling_impl_type != ScalingImplType.OVERRIDE and scaling_override is not None:
            raise Exception("Overriding scaling requires to set ScalingImplType to OVERRIDE explicitly.")
        if scaling_impl_type == ScalingImplType.OVERRIDE and scaling_override is None:
            raise Exception("Overriding scaling requires to pass a scaling impl module.")

        if scaling_impl_type == ScalingImplType.OVERRIDE and scaling_override is not None:
            scaling_impl = scaling_override

        elif scaling_impl_type == ScalingImplType.STATS \
                or scaling_impl_type == ScalingImplType.AFFINE_STATS \
                or scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
            stats_scaling = ParameterStatsScaling(stats_op=scaling_stats_op,
                                                  restrict_scaling_type=restrict_scaling_type,
                                                  tracked_parameter_list=tracked_parameter_list,
                                                  stats_input_view_shape_impl=scaling_stats_input_view_shape_impl,
                                                  stats_input_concat_dim=scaling_stats_input_concat_dim,
                                                  sigma=scaling_stats_sigma,
                                                  scaling_min_val=scaling_min_val,
                                                  stats_reduce_dim=scaling_stats_reduce_dim,
                                                  stats_output_shape=scaling_shape,
                                                  affine=scaling_impl_type == ScalingImplType.AFFINE_STATS)
            if scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
                if quant_type == QuantType.BINARY or quant_type == QuantType.TERNARY:
                    raise Exception("Parameter from stats scaling is currently not supported for binary/ternary")
                scaling_init = stats_scaling(zero_hw_sentinel).detach()
                scaling_impl = StandaloneScaling(scaling_init=scaling_init,
                                                 parameter_shape=scaling_shape,
                                                 restrict_scaling_type=restrict_scaling_type,
                                                 is_parameter=True,
                                                 scaling_min_val=scaling_min_val)
            else:
                scaling_impl = stats_scaling

        elif scaling_impl_type == ScalingImplType.CONST or scaling_impl_type == ScalingImplType.HE:
            if scaling_impl_type == ScalingImplType.HE:
                scaling_const = 0.0
                for param in tracked_parameter_list:  # takes average of He scaling over parameter list
                    two_dim_param = param.view(param.shape[0], -1)
                    scaling_const += math.sqrt(2.0 / two_dim_param.shape[1])
                scaling_const /= len(tracked_parameter_list)
            scaling_init = torch.tensor(scaling_const)
            scaling_impl = StandaloneScaling(scaling_init=scaling_init,
                                             parameter_shape=SCALING_SCALAR_SHAPE,
                                             restrict_scaling_type=restrict_scaling_type,
                                             is_parameter=False,
                                             scaling_min_val=None)
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
                    bit_width_impl = BitWidthConst(bit_width, restrict_bit_width_type)
                elif bit_width_impl_type == BitWidthImplType.PARAMETER:
                    bit_width_impl = BitWidthParameter(bit_width_init=bit_width,
                                                       restrict_bit_width_type=restrict_bit_width_type,
                                                       min_overall_bit_width=min_overall_bit_width,
                                                       max_overall_bit_width=max_overall_bit_width,
                                                       override_pretrained=override_pretrained_bit_width)
                else:
                    raise Exception("Bit width type {} not supported for weight quantization."
                                    .format(str(bit_width_impl_type)))
            else:
                bit_width_impl = bit_width_impl_override

            if bit_width_impl_type == BitWidthImplType.PARAMETER or \
                    bit_width_impl_type == BitWidthImplType.CONST and \
                    scaling_impl_type == ScalingImplType.PARAMETER_FROM_STATS:
                tensor_clamp_impl = TensorClamp()
            else:
                tensor_clamp_impl = TensorClampSte()

            float_to_int_impl = RestrictValue(restrict_value_type=RestrictValueType.INT,
                                              float_to_int_impl_type=FloatToIntImplType.ROUND,
                                              min_val=None)
            int_scaling_impl = IntScaling(narrow_range,
                                          signed=True,
                                          restrict_scaling_type=restrict_scaling_type)
            tensor_quant = RescalingIntQuant(narrow_range=narrow_range,
                                             signed=True,
                                             scaling_impl=scaling_impl,
                                             int_scaling_impl=int_scaling_impl,
                                             tensor_clamp_impl=tensor_clamp_impl,
                                             msb_clamp_bit_width_impl=bit_width_impl,
                                             float_to_int_impl=float_to_int_impl,
                                             runtime=False)
        else:
            raise Exception('Unsupported weight quantization: {} bit width, {} quantization.'
                            .format(bit_width, str(quant_type)))
    return tensor_quant


@docstrings.get_sectionsf('weight_quant_proxy')
class WeightQuantProxy(ParameterQuantProxy):
    """

    Parameters
    ----------

    bit_width
        The bit-width at which weights are quantized to. If `bit_width_impl_type` is set to ``PARAMETER``, this value is
        used for initialization. If `quant_type` is set to ``FP``, this value is ignored.
    quant_type
        Type of quantization. If set to ``FP``, no quantization is performed.
    narrow_range
        Restrict range of quantized values to a symmetrical interval around 0. For example, given `bit_width` set to
        8 and quant_type set to ``INT``, if `narrow_range` is set to ``True``, the range of quantized values is in
        ``[-127, 127]``; If set to ``False``, it's in ``[-128,127]``.
    restrict_scaling_type
        Type of restriction imposed on the values of the scaling factor of the quantized weights.
    scaling_const
        If `scaling_impl_type` is set to ``CONST``, this value is used as the scaling factor across all relevant
        dimensions. Ignored otherwise.
    scaling_stats_op
        Type of statistical operation performed for scaling, if required. If `scaling_impl_type` is set to ``STATS`` or
        ``AFFINE_STATS``, the operation is part of the compute graph and back-propagated through. If `scaling_impl_type`
        is set to ``PARAMETER_FROM_STATS``, the operation is used only for computing the initialization of the
        parameter, possibly across some dimensions. Ignored otherwise.
    scaling_impl_type
        Type of strategy adopted for scaling the quantized weights.
    scaling_stats_reduce_dim
        Dimension within the shape determined by `scaling_stats_input_view_shape_impl` along which `scaling_stats_op` is
        applied. If set to ``None``, scaling is assumed to be over the whole tensor. Ignored whenever `scaling_stats_op`
        is ignored.
    scaling_shape
        Shape of the scaling factor tensor. This is required to be broadcastable w.r.t. the weight tensor to scale.
    scaling_min_val
        Minimum value that the scaling factors can reach. This has precedence over anything else, including
        `scaling_const` when `scaling_impl_type` is set to ``CONST``. Useful in case of numerical instabilities.
        If set to None, no minimum is imposed.
    bit_width_impl_type
        Type of strategy adopted for precision at which the weights are quantized to when `quant_type` is set to
        ``INT``. Ignored otherwise.
    restrict_bit_width_type
        If `bit_width_impl_type` is set to ``PARAMETER`` and `quant_type` is set to ``INT``, this value constraints or
        relax the bit-width value that can be learned. Ignored otherwise.
    min_overall_bit_width
        If `bit_width_impl_type` is set to ``PARAMETER`` and `quant_type` is set to ``INT``, this value imposes a lower
        bound on the learned value. Ignored otherwise.
    max_overall_bit_width
        If `bit_width_impl_type` is set to ``PARAMETER`` and `quant_type` is set to ``INT``, this value imposes an upper
        bound on the learned value. Ignored otherwise.
    tracked_parameter_list_init
        Pytorch Parameter of which statistics are computed when `scaling_impl_type` is set to ``STATS``,
        ``AFFINE_STATS`` or ``PARAMETER_FROM_STATS``. This value initializes the list of parameters that are
        concatenated together when computing statistics.
    bit_width_impl_override
        Override the bit-width implementation with an implementation defined elsewhere. Accepts BitWidthConst or
        BitWidthParameter type of Modules. Useful for sharing the same learned bit-width between different layers.
    scaling_stats_input_view_shape_impl
        When `scaling_impl_type` is set to ``STATS``, ``AFFINE_STATS`` or ``PARAMETER_FROM_STATS``,
        this Module reshapes each tracked parameter before concatenating them together and computing their statistics.
    scaling_stats_input_concat_dim
        When `scaling_impl_type` is set to ``STATS``, ``AFFINE_STATS`` or ``PARAMETER_FROM_STATS``,
        this value defines the dimension along which the tracked parameters are concated after
        `scaling_stats_input_view_shape_impl` is called, but before statistics are taken.
    ternary_threshold
        Value to be used as a threshold when `quant_type` is set to ``TERNARY``. Ignored otherwise.
    scaling_stats_sigma
        Value to be used as sigma if `scaling_impl_type` is set to ``STATS``, ``AFFINE_STATS`` or
        ``PARAMETER_FROM_STATS`` and `scaling_stats_op` is set to ``AVE_SIGMA_STD`` or ``AVE_LEARN_SIGMA_STD``.
        Ignored otherwise. When `scaling_impl_type` is set to ``STATS`` or ``AFFINE_STATS``, and
        `scaling_stats_op` is set to ``AVE_LEARN_SIGMA_STD``, the value is used for initialization.
    override_pretrained_bit_width
        If set to ``True``, when loading a pre-trained model that includes a learned bit-width, the pre-trained value
        is ignored and replaced by the value specified by ``bit-width``.
    """

    def __init__(self,
                 bit_width: Optional[int],
                 quant_type: QuantType,
                 narrow_range: bool,
                 scaling_override: Optional[nn.Module],
                 restrict_scaling_type: RestrictValueType,
                 scaling_const: Optional[float],
                 scaling_stats_op: StatsOp,
                 scaling_impl_type: ScalingImplType,
                 scaling_stats_reduce_dim: Optional[int],
                 scaling_shape: Tuple[int, ...],
                 scaling_min_val: Optional[float],
                 bit_width_impl_type: Optional[BitWidthImplType],
                 restrict_bit_width_type: Optional[RestrictValueType],
                 min_overall_bit_width: Optional[int],
                 max_overall_bit_width: Optional[int],
                 tracked_parameter_list_init: torch.nn.Parameter,
                 bit_width_impl_override: Optional[Union[BitWidthConst, BitWidthParameter]],
                 scaling_stats_input_view_shape_impl: nn.Module,
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
                                              scaling_override=scaling_override,
                                              restrict_scaling_type=restrict_scaling_type,
                                              scaling_const=scaling_const,
                                              scaling_stats_op=scaling_stats_op,
                                              scaling_impl_type=scaling_impl_type,
                                              scaling_stats_reduce_dim=scaling_stats_reduce_dim,
                                              scaling_shape=scaling_shape,
                                              scaling_min_val=scaling_min_val,
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero_hw_sentinel = getattr(self, ZERO_HW_SENTINEL_NAME)
        out, scale, bit_width = self.tensor_quant(x, zero_hw_sentinel)
        reshaped_scale = scale.view(self.scale_output_shape)
        return out, reshaped_scale, bit_width

    def int_weight(self, x: torch.Tensor):
        zero_hw_sentinel = getattr(self, ZERO_HW_SENTINEL_NAME)
        quant_weight, scale, _ = self.tensor_quant(x, zero_hw_sentinel)
        quant_weight = quant_weight / scale
        quant_weight = round_ste(quant_weight)
        quant_weight = quant_weight.int()
        return quant_weight

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super(WeightQuantProxy, self)._load_from_state_dict(state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        if config.REINIT_WEIGHT_QUANT_ON_LOAD:
            self.re_init_tensor_quant()


class BiasQuantProxy(ParameterQuantProxy):

    def __init__(self,
                 quant_type: QuantType,
                 bit_width: Optional[int],
                 narrow_range: bool) -> None:
        super(BiasQuantProxy, self).__init__()
        self.scale_output_shape = OVER_BATCH_OVER_CHANNELS_SHAPE

        if quant_type == QuantType.FP:
            self.tensor_quant = None
        elif quant_type == QuantType.INT:
            tensor_clamp_impl = TensorClamp()
            float_to_int_impl = RestrictValue(restrict_value_type=RestrictValueType.INT,
                                              float_to_int_impl_type=FloatToIntImplType.ROUND,
                                              min_val=None)
            if bit_width is not None:
                bit_width_impl = BitWidthConst(bit_width, restrict_bit_width_type=RestrictValueType.INT)
                self.tensor_quant = PrescaledRestrictIntQuant(narrow_range=narrow_range,
                                                              signed=True,
                                                              tensor_clamp_impl=tensor_clamp_impl,
                                                              msb_clamp_bit_width_impl=bit_width_impl,
                                                              float_to_int_impl=float_to_int_impl)
                self.requires_input_bit_width = False
            else:
                self.tensor_quant = PrescaledIntQuant(narrow_range=narrow_range,
                                                      signed=True,
                                                      tensor_clamp_impl=tensor_clamp_impl,
                                                      float_to_int_impl=float_to_int_impl)
                self.requires_input_bit_width = True
        else:
            raise Exception('Quantization type {} not supported for bias quant.'
                            .format(str(quant_type)))

    def forward(self,
                x: Tensor,
                input_scale: Tensor,
                input_bit_width: Optional[Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        zero_hw_sentinel = getattr(self, ZERO_HW_SENTINEL_NAME)
        if self.tensor_quant is not None:

            if input_scale is None:
                raise Exception("Input scale can't be None when quantizing bias")
            input_scale = input_scale.view(-1)

            if self.requires_input_bit_width:  # bit width is defined outside
                if input_bit_width is None:
                    raise Exception("Input bit width can't be None when quantizing bias without a predefined bit width")
                out, output_scale, bias_bit_width = self.tensor_quant(x, input_scale, input_bit_width, zero_hw_sentinel)
            else:
                out, output_scale, bias_bit_width = self.tensor_quant(x, input_scale, zero_hw_sentinel)
            output_scale = output_scale.view(self.scale_output_shape)
            return out, output_scale, bias_bit_width
        else:
            return x, input_scale, input_bit_width


