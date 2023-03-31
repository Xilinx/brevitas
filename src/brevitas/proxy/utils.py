# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from torch.nn import Module


class ConvertRuntimeStatsToParameter:

    def __init__(self, restrict_scaling_impl: Module):
        self.restrict_scaling_impl = restrict_scaling_impl
        scaling_impl_postfix = 'fused_activation_quant_proxy.tensor_quant.scaling_impl'
        self.scaling_impl_postfix = scaling_impl_postfix
        self.runtime_stats_postfix = scaling_impl_postfix + '.runtime_stats'
        self.running_stats_postfix = scaling_impl_postfix + '.runtime_stats.running_stats'
        self.scaling_parameter_postfix = scaling_impl_postfix + '.value'

    def __call__(self, prefix, state_dict):
        running_stats_key = prefix + self.running_stats_postfix
        scaling_parameter_key = prefix + self.scaling_parameter_postfix
        # If it's retrained directly from statistics, i.e. there isn't a preexisting parameter
        if running_stats_key in state_dict and not scaling_parameter_key in state_dict:
            scaling_init = state_dict[running_stats_key]
            scaling_init = scaling_init.abs()
            # Preprocess scaling init, which is always in FP range, based on current restrictions
            scaling_init = self.restrict_scaling_impl.restrict_init_tensor(scaling_init)
            state_dict[scaling_parameter_key] = scaling_init
        # remove stats from dict
        for k in list(state_dict.keys()):
            if k.startswith(prefix + self.runtime_stats_postfix):
                del state_dict[k]
