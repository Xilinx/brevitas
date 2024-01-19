# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch import Tensor
from torch.nn import Parameter

from brevitas.proxy.parameter_quant import BiasQuantProxyFromInjector
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector


def compute_channel_view_shape(tensor: Tensor, channel_dim: int):
    broadcast_shape = [1] * len(tensor.size())
    broadcast_shape[channel_dim] = -1
    return tuple(broadcast_shape)


def mul_add_from_bn(bn_mean, bn_var, bn_eps, bn_weight, bn_bias):
    denom = torch.sqrt(bn_var + bn_eps)
    mul_factor = bn_weight / denom
    add_factor = -bn_mean * mul_factor + bn_bias
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
        layer.bias.data.mul_(mul_factor.view(out_ch_bias_shape))
        layer.bias.data.add_(add_factor.view(out_ch_bias_shape))
    else:
        layer.bias = Parameter(add_factor)
    if (hasattr(layer, 'weight_quant') and
            isinstance(layer.weight_quant, WeightQuantProxyFromInjector)):
        layer.weight_quant.init_tensor_quant()
    if (hasattr(layer, 'bias_quant') and isinstance(layer.bias_quant, BiasQuantProxyFromInjector)):
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


def check_tensors_same_ptr(tensor_list):
    pointers = []
    for t in tensor_list:
        if hasattr(t, 'data_ptr'):
            ptr = t.data_ptr()
            pointers.append(ptr)
        elif hasattr(t, 'value') and hasattr(t.value, 'data_ptr'):
            pointers.append(t.value.data_ptr())
        else:
            return False
    return all(p == pointers[0] for p in pointers)
