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


def mul_add_from_bn(bn_mean, bn_var, bn_eps, bn_weight, bn_bias, affine_only):
    mul_factor = bn_weight
    add_factor = bn_bias * torch.sqrt(bn_var + bn_eps)
    add_factor = add_factor - bn_mean * (bn_weight - 1.0)
    if not affine_only:
        mul_factor = mul_factor / torch.sqrt(bn_var + bn_eps)
        add_factor = add_factor - bn_mean
        add_factor = add_factor / torch.sqrt(bn_var + bn_eps)
    return mul_factor, add_factor


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