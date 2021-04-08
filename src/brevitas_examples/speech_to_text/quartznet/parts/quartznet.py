# Adapted from https://github.com/NVIDIA/NeMo/blob/r0.9/collections/nemo_asr/nemo_asr/parts/jasper.py
# Copyright (C) 2020 Xilinx (Giuseppe Franco)
# Copyright (C) 2019 NVIDIA CORPORATION.
#
# All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
from torch import Tensor
from .common import *
from brevitas.quant_tensor import QuantTensor
from brevitas.nn import QuantConv1d
from brevitas.nn.utils import mul_add_from_bn, rename_state_dict_by_postfix

jasper_activations = {
    "hardtanh": nn.Hardtanh,
    "relu": nn.ReLU,
    "selu": nn.SELU,
}


def init_weights(m, mode='xavier_uniform'):
    if isinstance(m, MaskedConv1d):
        init_weights(m.conv, mode)
    if isinstance(m, nn.Conv1d):
        if mode == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight, gain=1.0)
        elif mode == 'xavier_normal':
            nn.init.xavier_normal_(m.weight, gain=1.0)
        elif mode == 'kaiming_uniform':
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        elif mode == 'kaiming_normal':
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        else:
            raise ValueError("Unknown Initialization mode: {0}".format(mode))
    elif isinstance(m, nn.BatchNorm1d):
        if m.track_running_stats:
            m.running_mean.zero_()
            m.running_var.fill_(1)
            m.num_batches_tracked.zero_()
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2


class MaskedConv1d(nn.Module):
    __constants__ = ["use_conv_mask", "real_out_channels", "heads"]

    def __init__(self, in_channels, out_channels, kernel_size, scaling_per_channel, bit_width,
                 stride=1, padding=0, dilation=1, groups=1, heads=-1, bias=False, use_mask=True):
        super(MaskedConv1d, self).__init__()

        if not (heads == -1 or groups == in_channels):
            raise ValueError("Only use heads for depthwise convolutions")

        self.real_out_channels = out_channels
        if heads != -1:
            in_channels = heads
            out_channels = heads
            groups = heads
        self.conv = make_quantconv1d(in_channels, out_channels, kernel_size, bias=bias,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, scaling_per_channel=scaling_per_channel, bit_width=bit_width)
        self.channelwise_separable = (in_channels == out_channels) and (in_channels == groups)
        self.use_mask = use_mask
        self.heads = heads

    def get_seq_len(self, lens):
        return ((lens + 2 * self.conv.padding[0] - self.conv.dilation[0] * (
                self.conv.kernel_size[0] - 1) - 1) / self.conv.stride[0] + 1)

    def forward(self, x, lens):
        if self.use_mask:
            lens = lens.to(dtype=torch.long)
            max_len = x.size(2)
            mask = torch.arange(max_len).to(lens.device) \
                       .expand(len(lens), max_len) >= lens.unsqueeze(1)
            x = x.masked_fill(
                mask.unsqueeze(1).to(device=x.device), 0
            )
            # del mask
            lens = self.get_seq_len(lens)

        sh = x.shape
        if self.heads != -1:
            x = x.view(-1, self.heads, sh[-1])

        out = self.conv(x)

        if self.heads != -1:
            out = out.view(sh[0], self.real_out_channels, -1)

        return out, lens


class GroupShuffle(nn.Module):

    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()

        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape

        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])

        x = torch.transpose(x, 1, 2).contiguous()

        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])

        return x


class JasperBlock(nn.Module):
    __constants__ = ["conv_mask", "separable", "residual_mode", "res", "mconv"]

    def __init__(self, inplanes, planes, bit_width,
                 absolute_act_val,
                 activation_inner_scaling_per_output_channel, activation_other_scaling_per_output_channel,
                 weight_scaling_per_output_channel,
                 repeat=3, kernel_size=11, stride=1,
                 dilation=1, padding='same', dropout=0.2, activation=None,
                 residual=True, groups=1, separable=False,
                 heads=-1, normalization="batch",
                 norm_groups=1, residual_mode='add',
                 residual_panes=[], conv_mask=False, fused_bn=False):
        super(JasperBlock, self).__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")
        self.fused_bn = fused_bn
        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.conv_mask = conv_mask
        self.separable = separable
        self.residual_mode = residual_mode
        self.conv_module_to_merge = []
        inplanes_loop = inplanes
        conv = nn.ModuleList()
        self.norm_depthwise = nn.ModuleList()
        for _ in range(repeat - 1):
            if separable:
                self.norm_depthwise.extend(
                    [make_norm_scale(bit_width=bit_width,
                                     absolute_act_val=absolute_act_val,
                                     scaling_per_channel=activation_other_scaling_per_output_channel)]
                )
            conv.extend(self._get_conv_bn_layer(
                inplanes_loop,
                planes,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_val,
                groups=groups,
                heads=heads,
                separable=separable,
                normalization=normalization,
                norm_groups=norm_groups,
                bit_width=bit_width,
                scaling_per_channel=weight_scaling_per_output_channel))

            conv.extend(self._get_act_dropout_layer(
                drop_prob=dropout,
                activation=activation,
                channels=planes,
                bit_width=bit_width,
                absolute_act_val=absolute_act_val,
                scaling_per_channel=activation_inner_scaling_per_output_channel))

            inplanes_loop = planes

        if separable:
            self.norm_depthwise.extend(
                [make_norm_scale(bit_width=bit_width,
                                 absolute_act_val=absolute_act_val,
                                 scaling_per_channel=activation_other_scaling_per_output_channel)]
            )
        conv.extend(self._get_conv_bn_layer(
            inplanes_loop,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding=padding_val,
            groups=groups,
            heads=heads,
            separable=separable,
            normalization=normalization,
            norm_groups=norm_groups,
            bit_width=bit_width,
            scaling_per_channel=weight_scaling_per_output_channel))

        self.mconv = conv

        res_panes = residual_panes.copy()
        self.dense_residual = residual

        if residual:
            res_list = nn.ModuleList()
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                res_list.append(nn.ModuleList(self._get_conv_bn_layer(
                    ip,
                    planes,
                    kernel_size=1,
                    normalization=normalization,
                    norm_groups=norm_groups,
                    bit_width=bit_width,
                    scaling_per_channel=weight_scaling_per_output_channel)))
            self.res = res_list
            self.quant_normalization = make_norm_scale(
                bit_width=bit_width, absolute_act_val=absolute_act_val,
                scaling_per_channel=activation_other_scaling_per_output_channel)
        else:
            self.res = None
            self.quant_normalization = None

        self.mout = nn.Sequential(
            *self._get_act_dropout_layer(
                drop_prob=dropout,
                activation=activation,
                channels=inplanes_loop,
                absolute_act_val=absolute_act_val,
                scaling_per_channel=activation_other_scaling_per_output_channel,
                bit_width=bit_width)
        )

    def _get_conv(self, in_channels, out_channels, bit_width, scaling_per_channel, kernel_size=11,
                  stride=1, dilation=1, padding=0, bias=False,
                  groups=1, heads=-1, separable=False):
        use_mask = self.conv_mask
        if use_mask:
            return MaskedConv1d(in_channels, out_channels, kernel_size,
                                stride=stride,
                                dilation=dilation, padding=padding, bias=bias,
                                groups=groups, heads=heads,
                                use_mask=use_mask, scaling_per_channel=scaling_per_channel, bit_width=bit_width)
        else:
            return make_quantconv1d(in_channels, out_channels, kernel_size, stride=stride,
                                    dilation=dilation, padding=padding, groups=groups, bias=bias,
                                    scaling_per_channel=scaling_per_channel, bit_width=bit_width)

    def _get_conv_bn_layer(self, in_channels, out_channels, bit_width, scaling_per_channel, kernel_size=11,
                           stride=1, dilation=1, padding=0, bias=False,
                           groups=1, heads=-1, separable=False,
                           normalization="batch", norm_groups=1):
        if norm_groups == -1:
            norm_groups = out_channels

        if separable:
            layers = [
                self._get_conv(in_channels,
                               in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               dilation=dilation, padding=padding,
                               groups=in_channels, heads=heads, bias=bias,
                               scaling_per_channel=scaling_per_channel, bit_width=bit_width),
                self._get_conv(in_channels, out_channels, kernel_size=1,
                               stride=1,
                               dilation=1, padding=0, groups=groups, bias=bias,
                               scaling_per_channel=scaling_per_channel, bit_width=bit_width)
            ]
        else:
            layers = [
                self._get_conv(in_channels, out_channels, kernel_size=kernel_size,
                               scaling_per_channel=scaling_per_channel, bit_width=bit_width,
                               stride=stride, bias=bias,
                               dilation=dilation, padding=padding,
                               groups=groups)
            ]

        if normalization == "group":
            layers.append(nn.GroupNorm(
                num_groups=norm_groups, num_channels=out_channels))
        elif normalization == "instance":
            layers.append(nn.GroupNorm(
                num_groups=out_channels, num_channels=out_channels))
        elif normalization == "layer":
            layers.append(nn.GroupNorm(
                num_groups=1, num_channels=out_channels))
        elif normalization == "batch":
            if self.fused_bn:
                self.conv_module_to_merge.append(layers[-1])
                layers.append(nn.Identity())
            else:
                layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))
        else:
            raise ValueError(
                f"Normalization method ({normalization}) does not match"
                f" one of [batch, layer, group, instance].")

        if groups > 1:
            layers.append(GroupShuffle(groups, out_channels))
        return layers

    def _get_act_dropout_layer(self, channels, bit_width, absolute_act_val, scaling_per_channel, drop_prob=0.2, activation=None):
        if activation is None:
            raise Exception("Activation required")
        layers = [
            make_jasper_activation(activation, channels, bit_width=bit_width, absolute_act_val=absolute_act_val,
                                   scaling_per_channel=scaling_per_channel),
            nn.Dropout(p=drop_prob)
        ]
        return layers

    def forward(self, input_: Tuple[List[Tensor], Optional[Tensor]]):
        # type: (Tuple[List[Tensor], Optional[Tensor]]) -> Tuple[List[Tensor], Optional[Tensor]] # nopep8
        lens_orig = None
        xs = input_[0]
        if len(input_) == 2:
            xs, lens_orig = input_

        # compute forward convolutions
        out = xs[-1]
        count_norm = 0
        lens = lens_orig
        check_flag = False
        for i, l in enumerate(self.mconv):
            if isinstance(l, MaskedConv1d):
                out, lens = l(out, lens)
            else:
                out = l(out)
            if isinstance(l, (MaskedConv1d, QuantConv1d)):
                check_flag = check_flag or l.channelwise_separable
                if l.channelwise_separable:
                    out = self.norm_depthwise[count_norm](out)
                    if isinstance(out, QuantTensor):
                        out = out.value
                    count_norm += 1
        if check_flag:
            assert (len(self.norm_depthwise) == count_norm)

        # compute the residuals
        if self.res is not None:
            out = self.quant_normalization(out)
            if self.training:
                out = out.value
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if isinstance(res_layer, MaskedConv1d):
                        res_out, _ = res_layer(res_out, lens_orig)
                    else:
                        res_out = res_layer(res_out)
                res_out = self.quant_normalization(res_out)
                if self.training:
                    res_out = res_out.value
                if self.residual_mode == 'add':
                    out = out + res_out
                else:
                    out = torch.max(out, res_out)

        if isinstance(out, QuantTensor):
            out = out.value

        # compute the output
        out = self.mout(out)
        if self.res is not None and self.dense_residual:
            return xs + [out], lens

        return [out], lens

    def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs):
        if not self.conv_mask:
            rename_state_dict_by_postfix('conv.weight', 'weight', state_dict)
        if self.fused_bn:
            self.fuse_bn(state_dict, prefix)
        super(JasperBlock, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs)
        # fix pretrained models with declared but unused extra quantization layers
        extra_k = 'quant_normalization'
        is_prefix_to_fix = any([prefix == 'encoder.' + p for p in ["0.", "16.", "17."]])
        if is_prefix_to_fix:
            for i, k in enumerate(unexpected_keys):
                if extra_k in k:
                    del unexpected_keys[i]

    def fuse_bn(self, state_dict,
            prefix):
        index = 0
        flag = False
        keys_to_check = []
        keys_to_delete = []
        for k in state_dict.keys():
            if k.startswith(prefix):
                keys_to_check.append(k)
                if k.split('.')[-1] == 'running_mean':
                    flag = True


        if flag:
            for name in keys_to_check:
                prefix_long = name.split('.')[:-1]
                if name.split('.')[-1] == "running_mean":
                    bn_prefix = '.'.join(prefix_long)
                    module_number = int(prefix_long[-1])
                    conv_name = prefix_long[:-1] + [str(module_number-1)]
                    if self.conv_mask:
                        conv_name = conv_name + ['conv']
                    conv_name = '.'.join(conv_name)
                    conv_mod = self.conv_module_to_merge[index]
                    index = index + 1
                    bn_weight_key = '.'.join([bn_prefix, 'weight'])
                    bn_bias_key = '.'.join([bn_prefix, 'bias'])
                    bn_running_mean_key = '.'.join([bn_prefix, 'running_mean'])
                    bn_running_var_key = '.'.join([bn_prefix, 'running_var'])
                    bn_num_batches_traked_key = '.'.join([bn_prefix, 'num_batches_tracked'])
                    keys_to_delete = keys_to_delete + [bn_bias_key]
                    keys_to_delete = keys_to_delete + [bn_weight_key]
                    keys_to_delete = keys_to_delete + [bn_running_mean_key]
                    keys_to_delete = keys_to_delete + [bn_running_var_key]
                    keys_to_delete = keys_to_delete + [bn_num_batches_traked_key]
                    mul_factor, add_factor = mul_add_from_bn(
                        bn_mean=state_dict[bn_running_mean_key],
                        bn_var=state_dict[bn_running_var_key],
                        bn_eps=1e-03,
                        bn_weight=state_dict[bn_weight_key],
                        bn_bias=state_dict[bn_bias_key])
                    if isinstance(conv_mod, MaskedConv1d):
                        conv_mod = conv_mod.conv
                    conv_weight_key = conv_name + '.weight'
                    conv_bias_key = conv_name + '.bias'
                    result = state_dict[conv_weight_key] * mul_factor.view(-1, 1, 1)

                    state_dict[conv_weight_key] = result

                    if conv_mod.bias is not None and conv_bias_key in state_dict:
                        state_dict[conv_bias_key] += add_factor
                    elif conv_mod.bias is not None and not conv_bias_key in state_dict:
                        state_dict[conv_bias_key] = add_factor
                    else:
                        if torch.cuda.is_available():
                            add_factor = add_factor.cuda()
                        conv_mod.bias = nn.Parameter(add_factor)
                        # add it to the dict any to avoid missing key error
                        state_dict[conv_bias_key] = add_factor

                    # Get rid of statistics after using them
                else:
                    state_dict[name] = state_dict[name]
        for k in list(state_dict.keys()):
            if k in keys_to_delete:
                del state_dict[k]
        assert len(self.conv_module_to_merge) == index


