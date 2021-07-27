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

import torch
from torch.nn import Parameter

from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector, WeightQuantProxyFromInjector


def compute_channel_view_shape(tensor: torch.Tensor, channel_dim: int):
    broadcast_shape = [1] * len(tensor.size())
    broadcast_shape[channel_dim] = -1
    return tuple(broadcast_shape)


def mul_add_from_bn(bn_mean, bn_var, bn_eps, bn_weight, bn_bias):
    denom = torch.sqrt(bn_var + bn_eps)
    mul_factor = bn_weight / denom
    add_factor = - bn_mean * mul_factor + bn_bias
    return mul_factor, add_factor


def merge_bn(layer, bn, output_channel_dim=0):
    out = mul_add_from_bn(
        bn_mean=bn.running_mean,
        bn_var=bn.running_var,
        bn_eps=bn.eps,
        bn_weight=bn.weight.data.clone(),
        bn_bias=bn.bias.data.clone())
    mul_factor, add_factor = out
    out_ch_weight_shape = compute_channel_view_shape(layer.weight, output_channel_dim)
    layer.weight.data.mul_(mul_factor.view(out_ch_weight_shape))
    if layer.bias is not None:
        out_ch_bias_shape = compute_channel_view_shape(layer.bias, channel_dim=0)
        layer.bias.data.add_(add_factor.view(out_ch_bias_shape))
    else:
        layer.bias = Parameter(add_factor)
    if (hasattr(layer, 'weight_quant')
            and isinstance(layer.weight_quant, WeightQuantProxyFromInjector)):
        layer.weight_quant.init_tensor_quant()
    if (hasattr(layer, 'bias_quant')
            and isinstance(layer.bias_quant, BiasQuantProxyFromInjector)):
        layer.bias_quant.init_tensor_quant()


def rename_state_dict_by_prefix(old_prefix, new_prefix, state_dict):
    keys_map = {}
    for k in state_dict.keys():
        if k.startswith(old_prefix):
            new_key = new_prefix + k[len(old_prefix):]
            keys_map[k] = new_key
    for old_key in keys_map.keys():
        state_dict[keys_map[old_key]] = state_dict.pop(old_key)


def rename_state_dict_by_postfix(old_postfix, new_postfix, state_dict):
    keys_map = {}
    for k in state_dict.keys():
        if k.endswith(old_postfix):
            new_key = k[:len(k) - len(old_postfix)] + new_postfix
            keys_map[k] = new_key
    for old_key in keys_map.keys():
        state_dict[keys_map[old_key]] = state_dict.pop(old_key)