# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter

import brevitas
from brevitas import config
from brevitas.core.function_wrapper.misc import Identity
from brevitas.core.function_wrapper.ops_ste import ScalarClampMinSte
from brevitas.core.utils import StatelessBuffer
from brevitas.function.ops import max_int
from brevitas.quant_tensor import _unpack_quant_tensor
# Use custom implementation of kthvalue as work around to (b)float16 kernel limitations
from brevitas.utils.torch_utils import kthvalue

from .stats_wrapper import SCALAR_SHAPE

DEFAULT_STD_DEV_EPSILON = 1e-8


class NegativeMinOrZero(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim', 'keepdim']

    def __init__(
            self,
            stats_reduce_dim: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            keepdim: bool = False) -> None:
        super(NegativeMinOrZero, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.zero = StatelessBuffer(torch.tensor(0.0, dtype=dtype, device=device))
        self.keepdim = keepdim

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        if self.stats_reduce_dim is None:
            min_val = torch.min(x)
        else:
            min_val = torch.min(x, dim=self.stats_reduce_dim, keepdim=self.keepdim)[0]
        min_val = torch.clamp(min_val, max=self.zero())
        return min_val


class AbsPercentile(brevitas.jit.ScriptModule):
    __constants__ = ['q', 'stats_reduce_dim', 'keepdim']

    def __init__(
            self,
            high_percentile_q: float,
            stats_reduce_dim: Optional[int],
            percentile_q=None,
            keepdim: bool = False):
        super(AbsPercentile, self).__init__()
        if percentile_q is not None:
            raise RuntimeError("percentile_q is deprecated, please pass high_percentile_q.")
        assert high_percentile_q <= 100, "q has to be a percentage"
        self.q = high_percentile_q
        self.stats_reduce_dim = stats_reduce_dim
        self.keepdim = keepdim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            # k is 1-indexed, so round away from zero
            k = int(math.floor(.01 * self.q * x.numel() + 0.5))
            result = kthvalue(x.abs().view(-1), k)[0]
        else:
            # assuming x is two dimensional, get the other dimension
            assert len(x.size()) == 2, "Only 2-dim input is supported."
            other_dim = abs(self.stats_reduce_dim - 1)
            dim_slice = torch.narrow(x, dim=other_dim, start=0, length=1)
            # k is 1-indexed, so round away from zero
            k = int(math.floor(.01 * self.q * dim_slice.numel() + 0.5))
            result = kthvalue(x.abs(), k, dim=self.stats_reduce_dim, keepdim=self.keepdim)[0]
        return result


class NegativePercentileOrZero(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim', 'q', 'keepdim']

    def __init__(
            self,
            low_percentile_q,
            stats_reduce_dim: Optional[int] = None,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            keepdim: bool = False) -> None:
        super(NegativePercentileOrZero, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.q = low_percentile_q
        self.zero = StatelessBuffer(torch.tensor(0.0, dtype=dtype, device=device))
        self.keepdim = keepdim

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        if self.stats_reduce_dim is None:
            # k is 1-indexed, so round away from zero
            k = int(math.ceil(.01 * self.q * x.numel()))
            result = kthvalue(x.view(-1), k)[0]
        else:
            # assuming x is two dimensional, get the other dimension
            assert len(x.size()) == 2, "Only 2-dim input is supported."
            other_dim = abs(self.stats_reduce_dim - 1)
            dim_slice = torch.narrow(x, dim=other_dim, start=0, length=1)
            # k is 1-indexed, so round away from zero
            k = int(math.ceil(.01 * self.q * dim_slice.numel()))
            result = kthvalue(x, k, dim=self.stats_reduce_dim, keepdim=self.keepdim)[0]
        result = torch.clamp(result, max=self.zero())
        return result


class PercentileInterval(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim', 'low_q', 'high_q', 'keepdim']

    def __init__(
            self,
            low_percentile_q,
            high_percentile_q,
            stats_reduce_dim: Optional[int] = None,
            keepdim: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(PercentileInterval, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.low_q = low_percentile_q
        self.high_q = high_percentile_q
        self.keepdim = keepdim
        self.zero = StatelessBuffer(torch.tensor(0.0, dtype=dtype, device=device))

    @brevitas.jit.script_method
    def forward(self, x: Tensor) -> Tensor:
        if self.stats_reduce_dim is None:
            low_k = int(math.ceil(.01 * self.low_q * x.numel()))
            # k is 1-indexed, so round away from zero
            high_k = int(math.floor(.01 * self.high_q * x.numel() + 0.5))
            low_result = kthvalue(x.view(-1), low_k)[0]
            high_result = kthvalue(x.view(-1), high_k)[0]
        else:
            # assuming x is two dimensional, get the other dimension
            assert len(x.size()) == 2, "Only 2-dim input is supported."
            other_dim = abs(self.stats_reduce_dim - 1)
            dim_slice = torch.narrow(x, dim=other_dim, start=0, length=1)
            low_k = int(math.ceil(.01 * self.low_q * dim_slice.numel()))
            # k is 1-indexed, so round away from zero
            high_k = int(math.floor(.01 * self.high_q * dim_slice.numel() + 0.5))
            low_result = kthvalue(x, low_k, dim=self.stats_reduce_dim, keepdim=self.keepdim)[0]
            high_result = kthvalue(x, high_k, dim=self.stats_reduce_dim, keepdim=self.keepdim)[0]
        # We need to make sure the lower bound is not positive to align with zero-point statistics
        low_result = torch.clamp(low_result, max=self.zero())
        interval = high_result - low_result
        abs_interval = torch.abs(interval)
        return abs_interval


class AbsMax(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int] = None, keepdim: bool = False) -> None:
        super(AbsMax, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.keepdim = keepdim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            return torch.max(torch.abs(x))
        else:
            return torch.max(torch.abs(x), dim=self.stats_reduce_dim, keepdim=self.keepdim)[0]


class AbsMinMax(brevitas.jit.ScriptModule):
    __constants__ = ['stats_reduce_dim', 'keepdim']

    def __init__(
            self,
            stats_reduce_dim: Optional[int] = None,
            keepdim: bool = False,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(AbsMinMax, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim
        self.keepdim = keepdim
        self.zero = StatelessBuffer(torch.tensor(0.0, dtype=dtype, device=device))

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            max_val = torch.max(x)
            min_val = torch.min(x)
        else:
            max_val = torch.max(x, dim=self.stats_reduce_dim, keepdim=self.keepdim)[0]
            min_val = torch.min(x, dim=self.stats_reduce_dim, keepdim=self.keepdim)[0]
        # We need to make sure the lower bound is not positive to align with zero-point statistics
        min_val = torch.clamp(min_val, max=self.zero())
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
            std_dev_epsilon: float = DEFAULT_STD_DEV_EPSILON,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(MeanSigmaStd, self).__init__()
        self.impl = _MeanSigmaStdImpl(stats_reduce_dim, std_dev_epsilon)
        self.sigma = StatelessBuffer(torch.tensor(sigma, dtype=dtype, device=device))

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
            std_dev_epsilon: float = DEFAULT_STD_DEV_EPSILON,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None) -> None:
        super(MeanLearnedSigmaStd, self).__init__()
        self.impl = _MeanSigmaStdImpl(stats_reduce_dim, std_dev_epsilon)
        if stats_output_shape == SCALAR_SHAPE:
            self.value = Parameter(torch.tensor(sigma, dtype=dtype, device=device))
        else:
            self.value = Parameter(
                torch.full(stats_output_shape, sigma, dtype=dtype, device=device))

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        sigma = self.sigma.view(self.sigma.shape)  # trick to get a tensor type
        out = self.impl(x, sigma)
        return out

    def _load_from_state_dict(
            self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
            error_msgs):
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

    def __init__(self, signed, bit_width_impl, num_bins=1000 + 1, smoothing_eps=0.0001):
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


class L1Norm(brevitas.jit.ScriptModule):
    """ScriptModule implementation to collect per-channel L1 normalization stats
    for weight normalization-based quantization."""
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int] = None) -> None:
        super(L1Norm, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            # Need to be able to return the max per-channel L1 norm as a scalar
            raise NotImplementedError("L1 normalization is not supported per-tensor yet.")
        else:
            return x.norm(p=1, dim=self.stats_reduce_dim, keepdim=True)


class L2Norm(brevitas.jit.ScriptModule):
    """ScriptModule implementation to collect per-channel L2 normalization stats
    for weight normalization-based quantization."""
    __constants__ = ['stats_reduce_dim']

    def __init__(self, stats_reduce_dim: Optional[int] = None) -> None:
        super(L2Norm, self).__init__()
        self.stats_reduce_dim = stats_reduce_dim

    @brevitas.jit.script_method
    def forward(self, x: Tensor):
        if self.stats_reduce_dim is None:
            # Need to be able to return the max per-channel L2 norm as a scalar
            raise NotImplementedError("L2 normalization is not supported per-tensor yet.")
        else:
            return x.norm(p=2, dim=self.stats_reduce_dim, keepdim=True)


def _set_local_loss_mode(module, enabled):
    for m in module.modules():
        if hasattr(m, 'local_loss_mode'):
            m.local_loss_mode = enabled


def _set_observer_mode(module, enabled, previous_observer_mode):
    for m in module.modules():
        if hasattr(m, 'observer_only'):
            previous_observer_mode[m] = m.observer_only
            m.observer_only = enabled


def _restore_observer_mode(module, previous_observer_mode):
    for m in module.modules():
        if hasattr(m, 'observer_only'):
            m.observer_only = previous_observer_mode[m]


# If modules are offloaded, during local loss mode we need to re-allocate params after the search is complete
def _restore_params(module):
    for m in module.modules():
        if hasattr(m, 'local_loss_mode') and hasattr(m, 'allocate_params'):
            m.allocate_params(m)


class MSE(torch.nn.Module):
    # References:
    # https://github.com/cornell-zhang/dnn-quant-ocs/blob/master/distiller/quantization/clip.py
    # https://github.com/wimh966/outlier_suppression/blob/main/quant_transformer/quantization/observer.py

    def __init__(
            self,
            proxy_module,
            mse_init_op,
            inner_stats_input_view_shape_impl: torch.nn.Module,
            stats_reduce_dim: Optional[int] = None,
            mse_search_method: str = 'fibonacci',
            mse_iters: int = 20):
        super(MSE, self).__init__()
        self.mse_init_op = mse_init_op
        self.input_view_shape_impl = inner_stats_input_view_shape_impl
        self.proxy_forward = proxy_module.forward
        self.previous_observer_mode = dict()
        self.set_local_loss_mode = lambda enabled: _set_local_loss_mode(proxy_module, enabled)
        self.set_observer_mode = lambda enabled: _set_observer_mode(
            proxy_module, enabled, self.previous_observer_mode)
        self.restore_observer_mode = lambda: _restore_observer_mode(
            proxy_module, self.previous_observer_mode)
        self.restore_offload_param = lambda: _restore_params(proxy_module)
        self.internal_candidate = None
        self.num = mse_iters
        self.search_method = mse_search_method
        self.stats_reduce_dim = stats_reduce_dim
        self.local_loss_mode: bool = False

    def mse_loss_fn(self, x, quant_value):
        loss = torch.nn.functional.mse_loss(x, quant_value, reduction='none')
        if self.stats_reduce_dim is not None:
            # stats_reduce_dim applies to the permuted and reshaped tensor
            loss = self.input_view_shape_impl(loss)
            loss = torch.sum(loss, dim=self.stats_reduce_dim)
        else:
            loss = torch.sum(loss)
        return loss

    def evaluate_loss(self, x, candidate):
        self.internal_candidate = candidate
        # Set to local_loss_mode before calling the proxy
        self.set_local_loss_mode(True)
        self.set_observer_mode(False)
        quant_value = self.proxy_forward(x)
        quant_value = _unpack_quant_tensor(quant_value)
        loss = self.mse_loss_fn(x, quant_value)
        self.set_local_loss_mode(False)
        self.restore_observer_mode()
        return loss

    def mse_grid_search(self, xl, x):
        best_loss = torch.tensor(float('inf'), device=x.device, dtype=x.dtype)
        best_candidate = xl
        for i in range(2, self.num + 1):
            candidate = (xl * i).detach()
            loss = self.evaluate_loss(x, candidate)
            best_candidate = torch.where(loss < best_loss, candidate, best_candidate)
            best_loss = torch.min(loss, best_loss)
        return best_candidate

    def mse_fib_search(self, xl, xr, x):

        def fib_seq(n):
            if n <= 0:
                return [0]
            seq = [0, 1]
            while len(seq) <= n:
                next = seq[-1] + seq[-2]
                seq.append(next)
            return seq

        # vectorized variant of
        # https://indrag49.github.io/Numerical-Optimization/solving-one-dimensional-optimization-problems.html#fibonacci-search-method
        F = fib_seq(self.num)
        L0 = xr - xl
        Li = (F[self.num - 2] / F[self.num]) * L0
        for i in range(2, self.num + 1):
            x1 = torch.where(Li > L0 / 2, xr - Li, xl + Li)
            x2 = torch.where(Li > L0 / 2, xl + Li, xr - Li)
            f1, f2 = self.evaluate_loss(x, x1), self.evaluate_loss(x, x2)
            xr = torch.where(f1 <= f2, x2, xr)
            xl = torch.where(f1 >= f2, x1, xl)
            Li = (F[self.num - i] / F[self.num - (i - 2)]) * torch.where(f1 != f2, L0, xr - xl)
            L0 = xr - xl
        return torch.where(f1 <= f2, x1, x2)

    def mse_search(self, x):
        x_view = self.input_view_shape_impl(x)
        init = self.mse_init_op(x_view).detach()
        base = init / self.num
        if self.search_method == 'grid':
            best_candidate = self.mse_grid_search(base, x)
        elif self.search_method == 'fibonacci':
            best_candidate = self.mse_fib_search(base, init, x)
        else:
            raise ValueError(f"Search method {self.search_method} not supported.")
        # Save for evaluation by other modules (e.g. zp) invoking local loss mode
        self.internal_candidate = best_candidate
        self.restore_offload_param()
        return best_candidate

    def forward(self, x):
        if not self.local_loss_mode:
            with torch.no_grad():
                return self.mse_search(x)
        else:
            # This is invoked for the zero-point whenever scale is being optimized first
            if self.internal_candidate is None:
                x = self.input_view_shape_impl(x)
                self.internal_candidate = self.mse_init_op(x).detach()
            return self.internal_candidate


class HalfQuadraticOptimizerScale(torch.nn.Module):
    # References:
    # https://mobiusml.github.io/hqq_blog/
    # https://github.com/mobiusml/hqq?tab=readme-ov-file

    def __init__(
            self,
            proxy_module,
            hqo_init_op_scale,
            keepdim: bool,
            inner_stats_input_view_shape_impl: torch.nn.Module,
            scaling_min_val: Optional[float] = None,
            stats_reduce_dim: Optional[int] = None,
            int_scaling_impl=None,
            bit_width_impl=None,
            hqo_beta_scale: float = 1e5,
            hqo_kappa_scale: float = 1.01,
            hqo_lp_norm_scale: float = .7,
            hqo_iters_scale: int = 1000):
        super(HalfQuadraticOptimizerScale, self).__init__()
        self.hqo_init_op = hqo_init_op_scale
        self.input_view_shape_impl = inner_stats_input_view_shape_impl
        self.proxy_forward = proxy_module.forward
        self.previous_observer_mode = dict()
        self.set_local_loss_mode = lambda enabled: _set_local_loss_mode(proxy_module, enabled)
        self.set_observer_mode = lambda enabled: _set_observer_mode(
            proxy_module, enabled, self.previous_observer_mode)
        self.restore_observer_mode = lambda: _restore_observer_mode(
            proxy_module, self.previous_observer_mode)
        self.internal_candidate = None
        self.hqo_iters = hqo_iters_scale
        self.stats_reduce_dim = stats_reduce_dim
        self.local_loss_mode: bool = False

        self.beta = hqo_beta_scale
        self.kappa = hqo_kappa_scale
        self.lp_norm = hqo_lp_norm_scale

        self.int_scaling_impl = int_scaling_impl
        self.msb_clamp_bit_width_impl = bit_width_impl
        if scaling_min_val is not None and scaling_min_val != 0:
            self.clamp_min_ste = ScalarClampMinSte(scaling_min_val)
        else:
            self.clamp_min_ste = Identity()
        self.keepdim = keepdim

    def parameter_search(self, xl, x):
        best_loss = torch.tensor(float('inf'), device=x.device, dtype=x.dtype)
        candidate = xl
        best_candidate = candidate
        beta = self.beta
        with torch.no_grad():
            for i in range(0, self.hqo_iters):
                self.internal_candidate = candidate
                self.set_local_loss_mode(True)
                self.set_observer_mode(False)
                quant_tensor = self.proxy_forward(x).detach()
                self.set_local_loss_mode(False)
                self.restore_observer_mode()
                loss = torch.abs(quant_tensor.value - x).mean()

                best_candidate = torch.where(loss < best_loss, candidate, best_candidate)
                if loss >= best_loss:
                    break
                best_loss = torch.min(loss, best_loss)
                W_e = shrink_lp_op(x - quant_tensor.value, beta, self.lp_norm)
                zero_point = quant_tensor.zero_point
                num = self.input_view_shape_impl(x - W_e).detach()
                den = self.input_view_shape_impl(
                    torch.round(quant_tensor.value / quant_tensor.scale) - zero_point).detach()
                mask = (num != 0.) & (den != 0.)
                if self.stats_reduce_dim is None:
                    candidate = masked_median(num / den, mask)
                else:
                    candidate = masked_median(
                        num / den, mask, dim=self.stats_reduce_dim, keepdim=self.keepdim)
                candidate = candidate.type_as(self.internal_candidate)
                candidate = self.clamp_min_ste(candidate)
                bit_width = self.msb_clamp_bit_width_impl()
                int_threshold = self.int_scaling_impl(bit_width)
                candidate = candidate * int_threshold
                candidate[torch.isnan(candidate)] = self.internal_candidate[torch.isnan(candidate)]
                candidate[torch.isinf(candidate)] = self.internal_candidate[torch.isinf(candidate)]
                beta *= self.kappa
        return best_candidate

    def optimize(self, x):
        x_view = self.input_view_shape_impl(x)

        init = self.hqo_init_op(x_view).detach()
        best_candidate = self.parameter_search(init, x_view)

        # Save for evaluation by other modules (e.g. zp) invoking local loss mode
        self.internal_candidate = best_candidate.detach()
        torch.cuda.empty_cache()
        return best_candidate

    def forward(self, x):
        if not self.local_loss_mode:
            with torch.no_grad():
                return self.optimize(x)
        else:
            # This is invoked for the zero-point whenever scale is being optimized first
            if self.internal_candidate is None:
                x = self.input_view_shape_impl(x)
                self.internal_candidate = self.hqo_init_op(x).detach()
            return self.internal_candidate


class HalfQuadraticOptimizerZeroPoint(torch.nn.Module):
    # References:
    # https://mobiusml.github.io/hqq_blog/
    # https://github.com/mobiusml/hqq?tab=readme-ov-file

    def __init__(
            self,
            proxy_module,
            keepdim: bool,
            hqo_init_op_zp: torch.nn.Module,
            inner_stats_input_view_shape_impl: torch.nn.Module,
            stats_reduce_dim: Optional[int] = None,
            hqo_beta_zp: float = 1e0,
            hqo_kappa_zp: float = 1.01,
            hqo_lp_norm_zp: float = .5,
            hqo_iters_zp: int = 1000):
        super(HalfQuadraticOptimizerZeroPoint, self).__init__()
        self.hqo_init_op_zp = hqo_init_op_zp
        self.input_view_shape_impl = inner_stats_input_view_shape_impl
        self.proxy_forward = proxy_module.forward
        self.previous_observer_mode = dict()
        self.set_local_loss_mode = lambda enabled: _set_local_loss_mode(proxy_module, enabled)
        self.set_observer_mode = lambda enabled: _set_observer_mode(
            proxy_module, enabled, self.previous_observer_mode)
        self.restore_observer_mode = lambda: _restore_observer_mode(
            proxy_module, self.previous_observer_mode)
        self.internal_candidate = None
        self.stats_reduce_dim = stats_reduce_dim
        self.local_loss_mode: bool = False
        self.beta = hqo_beta_zp
        self.kappa = hqo_kappa_zp
        self.lp_norm = hqo_lp_norm_zp
        self.hqo_iters = hqo_iters_zp
        self.keepdim = keepdim

    def parameter_search(self, xl, x):
        best_loss = torch.tensor(float('inf'), device=x.device, dtype=x.dtype)
        candidate = xl
        best_candidate = candidate
        with torch.no_grad():
            for i in range(0, self.hqo_iters):
                self.internal_candidate = candidate
                self.set_local_loss_mode(True)
                self.set_observer_mode(False)
                quant_tensor = self.proxy_forward(x).detach()
                self.set_local_loss_mode(False)
                self.restore_observer_mode()
                qt_value = self.input_view_shape_impl(quant_tensor.value)
                qt_scale = self.input_view_shape_impl(quant_tensor.scale)
                qt_zp = self.input_view_shape_impl(quant_tensor.zero_point)
                qt_int = qt_value / qt_scale + qt_zp
                loss = torch.abs(qt_value - x).mean()
                best_candidate = torch.where(loss < best_loss, candidate, best_candidate)
                if loss >= best_loss:
                    break
                best_loss = torch.min(loss, best_loss)
                W_e = shrink_lp_op(x - qt_value, self.beta, self.lp_norm)

                # Compared to the original formulation, the value we're looking for is:
                # - scaled by qt_scale
                # - opposite sign
                val = self.input_view_shape_impl((x - W_e) - qt_int * qt_scale)

                if self.stats_reduce_dim is None:
                    candidate = torch.mean(val)
                else:
                    candidate = torch.mean(val, dim=self.stats_reduce_dim, keepdim=self.keepdim)
                self.beta *= self.kappa
        return best_candidate

    def optimize(self, x):
        x_view = self.input_view_shape_impl(x)

        init = self.hqo_init_op_zp(x_view).detach()

        best_candidate = self.parameter_search(init, x)

        # Save for evaluation by other modules (e.g. zp) invoking local loss mode
        self.internal_candidate = best_candidate.detach()
        torch.cuda.empty_cache()
        return best_candidate

    def forward(self, x):
        if not self.local_loss_mode:
            with torch.no_grad():
                return self.optimize(x)
        else:
            # This is invoked for the zero-point whenever scale is being optimized first
            if self.internal_candidate is None:
                x = self.input_view_shape_impl(x)
                self.internal_candidate = self.hqo_init_op_zp(x).detach()
            return self.internal_candidate


def masked_median(x, mask, dim=None, keepdim=False):
    """Compute the median of tensor x along dim, ignoring values where mask is False.
    x and mask need to be broadcastable.

    Args:
        x (Tensor): Tensor to compute median of.
        mask (BoolTensor): Same shape as x with True where x is valid and False
            where x should be masked. Mask should not be all False in any column of
            dimension dim to avoid NaNs from zero division.
        dim (int, optional): Dimension to take median of. Defaults to 0.

    Returns:
        Tensor: Same shape as x, except dimension dim reduced.
    """
    # uncomment this assert for safety but might impact performance
    # assert (
    #     mask.sum(dim=dim).ne(0).all()
    # ), "mask should not be all False in any column, causes zero division"
    x_nan = x.float().masked_fill(~mask, float("nan"))
    if dim is None:
        x_median = x_nan.nanmedian()
    else:
        x_median, _ = x_nan.nanmedian(dim=dim, keepdim=keepdim)
    return x_median


# Shrinking operator
def shrink_lp_op(x: Tensor, beta: float, lp_norm: float) -> Tensor:
    if lp_norm == 1:
        return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
    else:
        return torch.sign(x) * torch.nn.functional.relu(
            torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x), lp_norm - 1))
