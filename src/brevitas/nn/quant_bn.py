# Copyright (c) 2019-     Xilinx, Inc              (Giuseppe Franco)
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

from abc import ABC
from typing import Optional

import brevitas.config as config
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from .quant_scale_bias import QuantScaleBias
from .utils import mul_add_from_bn
from .quant_layer import WeightQuantType, BiasQuantType, ActQuantType


class _BatchNormToQuantScaleBias(QuantScaleBias, ABC):

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'weight'
        bias_key = prefix + 'bias'
        running_mean_key = prefix + 'running_mean'
        running_var_key = prefix + 'running_var'
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if running_mean_key in state_dict and running_var_key in state_dict:
            weight_init, bias_init = mul_add_from_bn(
                bn_bias=state_dict[bias_key],
                bn_weight=state_dict[weight_key],
                bn_mean=state_dict[running_mean_key],
                bn_var=state_dict[running_var_key],
                bn_eps=self.eps)
            self.weight.data = weight_init
            self.bias.data = bias_init
            del state_dict[bias_key]
            del state_dict[weight_key]
            del state_dict[running_mean_key]
            del state_dict[running_var_key]
            del state_dict[num_batches_tracked_key]
        super(_BatchNormToQuantScaleBias, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if config.IGNORE_MISSING_KEYS and bias_key in missing_keys:
            missing_keys.remove(bias_key)
        if config.IGNORE_MISSING_KEYS and weight_key in missing_keys:
            missing_keys.remove(weight_key)
        if num_batches_tracked_key in unexpected_keys:
            unexpected_keys.remove(num_batches_tracked_key)


class BatchNorm1dToQuantScaleBias(_BatchNormToQuantScaleBias):

    def __init__(
            self,
            num_features,
            eps: float = 1e-5,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        super(BatchNorm1dToQuantScaleBias, self).__init__(
            num_features,
            bias=True,
            runtime_shape=(1, -1, 1),
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self.eps = eps


class BatchNorm2dToQuantScaleBias(_BatchNormToQuantScaleBias):

    def __init__(
            self,
            num_features,
            eps: float = 1e-5,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        super(BatchNorm2dToQuantScaleBias, self).__init__(
            num_features,
            bias=True,
            runtime_shape=(1, -1, 1, 1),
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self.eps = eps
