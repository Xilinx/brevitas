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

from typing import Optional, Union

import math
import torch
from torch.nn import Linear, Module
from torch.nn.functional import linear

from brevitas.core.bit_width import BitWidthParameter, BitWidthConst, BitWidthImplType
from brevitas.core.quant import QuantType, IdentityQuant
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType, SCALING_SCALAR_SHAPE
from brevitas.core.stats import StatsInputViewShapeImpl
from brevitas.core.stats import StatsOp
from brevitas.function.ops import ceil_ste, max_uint
from brevitas.proxy.parameter_quant import WeightQuantProxy, BiasQuantProxy, WeightReg
from brevitas.config import docstrings
from .quant_layer import QuantLayer, SCALING_MIN_VAL

import brevitas.onnx as bo
from brevitas.onnx.onnx_custom_ops import QuantizedLinearPlaceholderFunction

__all__ = ['QuantLinear']


@docstrings.dedent
class QuantLinear(QuantLayer, Linear):
    """

        Parameters
        ----------

        %(weight_quant_proxy.parameters_with_prefix)s
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 bias_quant_type: QuantType = QuantType.FP,
                 bias_narrow_range: bool = False,
                 bias_bit_width: int = None,
                 weight_quant_override: WeightQuantProxy = None,
                 weight_quant_type: QuantType = QuantType.FP,
                 weight_narrow_range: bool = False,
                 weight_bit_width_impl_override: Union[BitWidthParameter, BitWidthConst] = None,
                 weight_bit_width_impl_type: BitWidthImplType = BitWidthImplType.CONST,
                 weight_restrict_bit_width_type: RestrictValueType = RestrictValueType.INT,
                 weight_bit_width: int = 32,
                 weight_min_overall_bit_width: Optional[int] = 2,
                 weight_max_overall_bit_width: Optional[int] = None,
                 weight_scaling_override: Optional[Module] = None,
                 weight_scaling_impl_type: ScalingImplType = ScalingImplType.STATS,
                 weight_scaling_const: Optional[float] = None,
                 weight_scaling_stats_op: StatsOp = StatsOp.MAX,
                 weight_scaling_per_output_channel: bool = False,
                 weight_scaling_min_val: float = SCALING_MIN_VAL,
                 weight_ternary_threshold: float = 0.5,
                 weight_restrict_scaling_type: RestrictValueType = RestrictValueType.LOG_FP,
                 weight_scaling_stats_sigma: float = 3.0,
                 weight_override_pretrained_bit_width: bool = False,
                 compute_output_scale: bool = False,
                 compute_output_bit_width: bool = False,
                 return_quant_tensor: bool = False) -> None:
        QuantLayer.__init__(self,
                            compute_output_scale=compute_output_scale,
                            compute_output_bit_width=compute_output_bit_width,
                            return_quant_tensor=return_quant_tensor)
        Linear.__init__(self,
                        in_features=in_features,
                        out_features=out_features,
                        bias=bias)
        # save a copy of args passed constructor, used to determine whether
        # the quantization config is exportable to something FINN supports
        self.init_args = locals()
        if weight_quant_type == QuantType.FP and compute_output_bit_width:
            raise Exception("Computing output bit width requires enabling quantization")
        if bias_quant_type != QuantType.FP and not (compute_output_scale and compute_output_bit_width):
            raise Exception("Quantizing bias requires to compute output scale and output bit width")

        self.per_elem_ops = 2 * in_features
        self.weight_reg = WeightReg()

        if weight_quant_override is not None:
            self.weight_quant = weight_quant_override
            self.weight_quant.add_tracked_tensor(self.weight)
        else:
            weight_scaling_stats_input_concat_dim = 1
            if weight_scaling_per_output_channel:
                weight_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_OUTPUT_CHANNELS
                weight_scaling_shape = (self.out_features, 1)
                weight_scaling_stats_reduce_dim = 1
            else:
                weight_stats_input_view_shape_impl = StatsInputViewShapeImpl.OVER_TENSOR
                weight_scaling_shape = SCALING_SCALAR_SHAPE
                weight_scaling_stats_reduce_dim = None

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
                                         narrow_range=bias_narrow_range,
                                         bit_width=bias_bit_width)

    def get_exportable_quantization_type(self):
        # Brevitas provides a wide range of possibilities for quantization,
        # but FINN only supports a subset. Here we test the quantization
        # config to see if it's something that FINN would understand.
        # TODO: the checks below are overly conservative, relax these.
        # alternatively, create specialized subclasses and only provide export
        # flows for those.
        ia = self.init_args
        if (
            ia["bias"] == False and
            ia["weight_quant_type"] == QuantType.BINARY and
            ia["weight_bit_width"] == 1 and
            ia["weight_bit_width_impl_type"] == BitWidthImplType.CONST and
            ia["weight_scaling_stats_op"] == StatsOp.AVE and
            ia["weight_scaling_stats_sigma"] == 0.001 and
            ia["weight_quant_override"] == None and
            ia["weight_narrow_range"] == False and
            ia["weight_bit_width_impl_override"] == None and
            ia["weight_bit_width_impl_type"] == BitWidthImplType.CONST and
            ia["weight_restrict_bit_width_type"] == RestrictValueType.INT and
            ia["weight_min_overall_bit_width"] == 2 and
            ia["weight_max_overall_bit_width"] == None and
            ia["weight_scaling_impl_type"] == ScalingImplType.STATS and
            ia["weight_scaling_min_val"] == SCALING_MIN_VAL and
            ia["weight_ternary_threshold"] == 0.5 and
            ia["weight_restrict_scaling_type"] == RestrictValueType.LOG_FP and
            ia["weight_override_pretrained_bit_width"] == False and
            ia["compute_output_scale"] == False and
            ia["compute_output_bit_width"] == False and
            ia["return_quant_tensor"] == False
            ):
            return "BIPOLAR"
        else:
            raise Exception("Unsupported config combination for export")

    @QuantLayer.export_mode.setter
    def export_mode(self, value):
        self._export_mode = value
        # create completely detached prequantized tensors for export
        # calling these in forward() causes the ops to be included in the graph
        # as dead-end nodes. note: this might be fixed in PyTorch 1.2.0 and
        # if so this workaround prepare_for_export is not necessary.
        self.export_int_weight = self.int_weight.detach()
        self.export_quant_weight_scale = self.quant_weight_scale.detach()

    @property
    def int_weight(self):
        if isinstance(self.weight_quant.tensor_quant, IdentityQuant):
            raise Exception("Can't generate int weight without quantization enabled")
        return self.weight_quant.int_weight(self.weight)

    @property
    def quant_weight_scale(self):
        if isinstance(self.weight_quant.tensor_quant, IdentityQuant):
            raise Exception("Can't generate scaling factor without quantization enabled")
        zero_hw_sentinel = self.weight_quant.zero_hw_sentinel
        return self.weight_quant.tensor_quant.scaling_impl(zero_hw_sentinel)

    def forward(self, input):
        if self.export_mode:
            export_qnt_type = self.get_exportable_quantization_type()
            # TODO what to do about the scale here? per out ch scaling also
            return QuantizedLinearPlaceholderFunction.apply(self.export_int_weight, input, export_qnt_type, self.out_features)
        else:
            output_scale = None
            output_bit_width = None

            input, input_scale, input_bit_width = self.unpack_input(input)

            quant_weight, quant_weight_scale, quant_weight_bit_width = self.weight_quant(self.weight)
            quant_weight = self.weight_reg(quant_weight)

            if self.compute_output_bit_width:
                output_bit_width = self.max_output_bit_width(input_bit_width, quant_weight_bit_width)
            if self.compute_output_scale:
                output_scale = input_scale * quant_weight_scale

            if self.bias is not None:
                quant_bias = self.bias_quant(self.bias, output_scale, output_bit_width)
                output = linear(input, quant_weight, quant_bias)
            else:
                output = linear(input, quant_weight, None)
            return self.pack_output(output, output_scale, output_bit_width)

    def max_output_bit_width(self, input_bit_width, weight_bit_width):
        max_input_val = max_uint(bit_width=input_bit_width, narrow_range=False)
        max_fc_val = self.weight_quant.tensor_quant.int_quant.max_uint(weight_bit_width)
        max_output_val = max_input_val * max_fc_val * self.in_features
        output_bit_width = ceil_ste(torch.log2(max_output_val))
        return output_bit_width
