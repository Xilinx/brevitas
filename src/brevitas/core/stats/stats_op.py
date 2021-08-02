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


from typing import Optional, Tuple
import math

import torch
from torch import Tensor
from torch.nn import Parameter

import brevitas
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops import max_int
from brevitas import config
from .stats_wrapper import SCALAR_SHAPE


DEFAULT_STD_DEV_EPSILON = 1e-8


class NegativeMinOrZero(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int] = None) -> None:
        super(NegativeMinOrZero, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.zero = StatelessBuffer(torch.tensor(0.0))

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        if self.stats_reduce_dim is None:
            min_val = torch.min(x)
        else:
            min_val = torch.min(x, dim=self.stats_reduce_dim)[0]
        min_val = torch.where(min_val <= self.zero(), min_val, self.zero())
        return min_val


class AbsPercentile(brevitas.jit.ScriptModule):
    __constants__ = ['q', 'stats_reduce_dim']

    def __init__(self, percentile_q: float, stats_reduce_dim: Optional[int]):
        super(AbsPercentile, self).__init__()
        assert percentile_q <= 100, "q has to be a percentage"
        self.q = percentile_q
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            # k is 1-indexed, so round away from zero
            k = int(math.floor(.01 * self.q * x.numel() + 0.5))
            result = x.abs().view(-1).kthvalue(k).values
        else:
            # assuming x is two dimensional, get the other dimension
            other_dim = abs(self.stats_reduce_dim - 1)
            dim_slice = torch.narrow(x, dim=other_dim, start=0, length=1)
            # k is 1-indexed, so round away from zero
            k = int(math.floor(.01 * self.q * dim_slice.numel() + 0.5))
            result = x.abs().kthvalue(k, dim=self.stats_reduce_dim).values
        return result


class AbsMax(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int] = None) -> None:
        super(AbsMax, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            return torch.max(torch.abs(x))
        else:
            return torch.max(torch.abs(x), dim=self.stats_reduce_dim)[0]


class AbsMinMax(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int] = None) -> None:
        super(AbsMinMax, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            return torch.abs(torch.max(x) - torch.min(x))
        else:
            max_val = torch.max(x, dim=self.stats_reduce_dim)[0]
            min_val = torch.min(x, dim=self.stats_reduce_dim)[0]
            return torch.abs(max_val - min_val)


class AbsMaxAve(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: int) -> None:
        super(AbsMaxAve, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        return torch.mean(torch.max(torch.abs(x), dim=self.stats_reduce_dim)[0])


class AbsMaxL2(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: int) -> None:
        super(AbsMaxL2, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        per_channel_max = torch.max(torch.abs(x), dim=self.stats_reduce_dim)[0]
        out = torch.norm(per_channel_max, p=2)
        out = out / math.sqrt(per_channel_max.view(-1).shape[0])
        return out


class AbsAve(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int] = None) -> None:
        super(AbsAve, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            return torch.mean(torch.abs(x))
        else:
            return torch.mean(torch.abs(x), dim=self.stats_reduce_dim)


class MeanSigmaStd(brevitas.jit.ScriptModule):

    def __init__(
            self,
            sigma: float,
            stats_reduce_dim: Optional[int] = None,
            std_dev_epsilon: float = DEFAULT_STD_DEV_EPSILON) -> None:
        super(MeanSigmaStd, self).__init__()
        self.impl = _MeanSigmaStdImpl(stats_reduce_dim, std_dev_epsilon)
        self.sigma = StatelessBuffer(torch.tensor(sigma))

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        sigma = self.sigma()
        out = self.impl(x, sigma)
        return out


class _MeanSigmaStdImpl(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim', 'output_shape', 'epsilon']

    def __init__(
            self,
            stats_reduce_dim: Optional[int] = None,
            std_dev_epsilon: float = DEFAULT_STD_DEV_EPSILON) -> None:
        super(_MeanSigmaStdImpl, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.epsilon = std_dev_epsilon

    @brevitas.jit.script_method
    def forward(self, x: Tensor, sigma: Tensor):
        abs_val = torch.abs(x)
        if self.stats_reduce_dim is None:
            mean_val = torch.mean(abs_val)
            std_val = torch.sqrt(torch.var(abs_val) + self.epsilon)
        else:
            mean_val = torch.mean(torch.abs(x), dim=self.stats_reduce_dim)
            std_val = torch.sqrt(torch.var(abs_val, dim=self.stats_reduce_dim) + self.epsilon)
            mean_val = mean_val.view(-1)
            std_val = std_val.view(-1)
        return mean_val + sigma * std_val


class MeanLearnedSigmaStd(brevitas.jit.ScriptModule):

    def __init__(
            self,
            sigma: float,
            stats_output_shape: Tuple[int, ...],
            stats_reduce_dim: Optional[int] = None,
            std_dev_epsilon: float = DEFAULT_STD_DEV_EPSILON) -> None:
        super(MeanLearnedSigmaStd, self).__init__()
        self.impl = _MeanSigmaStdImpl(stats_reduce_dim, std_dev_epsilon)
        if stats_output_shape == SCALAR_SHAPE:
            self.value = Parameter(torch.tensor(sigma))
        else:
            self.value = Parameter(torch.full(stats_output_shape, sigma))

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        sigma = self.sigma.view(self.sigma.shape)  # trick to get a tensor type
        out = self.impl(x, sigma)
        return out

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        value_key = prefix + 'sigma'
        retrocomp_value_key = prefix + 'learned_sigma'
        if retrocomp_value_key in state_dict:  # retrocompatibility
            state_dict[value_key] = state_dict.pop(retrocomp_value_key)
        super(MeanLearnedSigmaStd, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        sigma_key = prefix + 'sigma'
        if config.IGNORE_MISSING_KEYS and sigma_key in missing_keys:
            missing_keys.remove(sigma_key)


class KLMinimizerThreshold(torch.nn.Module):
    """
    Based on:
    https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
    """

    def __init__(self, signed, bit_width_impl, num_bins = 1000 + 1, smoothing_eps=0.0001):
        super(KLMinimizerThreshold, self).__init__()
        self.num_bins = num_bins
        self.smoothing_eps = smoothing_eps
        self.signed = signed
        self.bit_width_impl = bit_width_impl
        self.absmax_impl = AbsMax()

    def smooth_normalize_distribution(self, p, eps):
        is_zeros = (p == 0).float()
        n_zeros = is_zeros.sum()
        n_nonzeros = torch.numel(p) - n_zeros
        if not n_nonzeros:
            return None
        eps1 = eps * n_zeros / n_nonzeros
        hist = p.float()
        hist += eps * is_zeros + (-eps1) * n_nonzeros
        dist = torch.distributions.categorical.Categorical(logits=hist)
        return dist

    def forward(self, x: Tensor):
        absmax = self.absmax_impl(x)
        bit_width = self.bit_width_impl()
        num_quantized_bins = max_int(self.signed, False, bit_width).int()
        thresholds = torch.zeros(self.num_bins // 2 + 1 - num_quantized_bins // 2, device=x.device)
        divergence = torch.zeros_like(thresholds)
        quantized_bins = torch.zeros(num_quantized_bins, device=x.device)
        hist = torch.histc(x, bins=self.num_bins, min=-absmax, max=absmax).int()
        hist_edges = torch.linspace(-absmax, absmax, self.num_bins + 1)
        for i in range(num_quantized_bins // 2, self.num_bins // 2 + 1):
            p_bin_idx_start = self.num_bins // 2 - i
            p_bin_idx_stop = self.num_bins // 2 + i + 1
            thresholds[i - num_quantized_bins // 2] = hist_edges[p_bin_idx_stop]
            sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]
            p = sliced_nd_hist.clone()
            left_outlier_count = torch.sum(hist[0:p_bin_idx_start])
            p[0] += left_outlier_count
            right_outlier_count = torch.sum(hist[p_bin_idx_stop:])
            p[-1] += right_outlier_count
            is_nonzeros = (sliced_nd_hist != 0).float()
            num_merged_bins = torch.numel(p) // num_quantized_bins
            for j in range(num_quantized_bins):
                start = j * num_merged_bins
                stop = start + num_merged_bins
                quantized_bins[j] = sliced_nd_hist[start:stop].sum()
            quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
            q = torch.zeros_like(p, dtype=torch.float32, device=x.device)
            for j in range(num_quantized_bins):
                start = j * num_merged_bins
                if j == num_quantized_bins - 1:
                    stop = -1
                else:
                    stop = start + num_merged_bins
                norm = is_nonzeros[start:stop].sum()
                if norm != 0:
                    q[start:stop] = quantized_bins[j] / norm
            q[sliced_nd_hist == 0] = 0.
            p = self.smooth_normalize_distribution(p, self.smoothing_eps)
            q = self.smooth_normalize_distribution(q, self.smoothing_eps)
            if q is None:
                divergence[i - num_quantized_bins // 2] = float('inf')
            else:
                divergence[i - num_quantized_bins // 2] = torch.distributions.kl.kl_divergence(p, q)
        min_divergence_idx = torch.argmin(divergence)
        opt_threshold = thresholds[min_divergence_idx]
        return opt_threshold