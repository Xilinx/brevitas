# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from brevitas.function.ops_ste import ceil_ste
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


def calculate_min_accumulator_bit_width(
        input_bit_width: Tensor,
        input_is_signed: bool,
        weight_max_l1_norm: Optional[Tensor] = None,
        weight_bit_width: Optional[Tensor] = None,
        n_elements: Optional[Tensor] = None,
        min_val: Optional[float] = 1e-10):
    """Using the closed-form bounds on accumulator bit-width as derived in `Quantized Neural Networks for Low-Precision Accumulation with Guaranteed Overflow
    Avoidance` by I. Colbert, A. Pappalardo, and J. Petri-Koenig. This function returns the minimum accumulator bit-width that can be used without risk of
    overflow. It supports both the data-type bound as well as the weight-level bound.

    Args:
        input_bit_width (Tensor): the bit-width of the inputs to the layer.
        input_is_signed (bool): calculate statistics for normalizing weight parameter.
        weight_max_l1_norm (Tensor): the maximum per-channel l1-norm of the weights.
        weight_bit_width (Tensor): the bit-width of the weights to the layer.
        n_elements (Tensor): the number of elements in the dot product.
        min_val (float): the minimum value used for the l1-norm, used to avoid log2(0). Default: 1e-8.

    Example (data-type bound):
    >> acc_bit_width = calculate_min_accumulator_bit_width(input_bit_width, input_is_signed, weight_bit_width, n_elements)

    Example (weight-level bound):
    >> acc_bit_width = calculate_min_accumulator_bit_width(input_bit_width, input_is_signed, weight_max_l1_norm)
    """
    input_is_signed = float(input_is_signed)
    # if the l1-norm of the weights is specified, then use the weight-level bound
    if weight_max_l1_norm is not None:
        assert isinstance(weight_max_l1_norm, (float, Tensor)), "The l1-norm of the weights needs to be a float or a torch.Tensor instance."
        if isinstance(weight_max_l1_norm, Tensor):
            assert weight_max_l1_norm.numel() == 1, "The minimum accumulator bit-width calculation currently only supports scalars."
        weight_max_l1_norm = torch.clamp_min(weight_max_l1_norm, min_val)
        input_is_signed = float(input_is_signed)
        alpha = torch.log2(weight_max_l1_norm) + input_bit_width - input_is_signed
    # else use the data-type bound
    else:
        assert isinstance(weight_bit_width, (float, Tensor)), "If weight_max_l1_norm is un-specified, weight_bit_width needs to be specified."
        assert isinstance(n_elements, (float, Tensor)), "If weight_max_l1_norm is un-specified, n_elements needs to be specified."
        if isinstance(n_elements, Tensor):
            assert n_elements.numel() == 1, "The minimum accumulator bit-width calculation currently only supports scalars."
        assert n_elements > 0, "There needs to be at least one element considered in this evaluation."
        alpha = torch.log2(n_elements) + input_bit_width + weight_bit_width - input_is_signed - 1.
    phi = lambda x: torch.log2(1. + pow(2., -x))
    min_bit_width = alpha + phi(alpha) + 1.
    min_bit_width = ceil_ste(min_bit_width)
    return min_bit_width  # returns the minimum accumulator that can be used without risk of overflow
