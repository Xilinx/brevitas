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
