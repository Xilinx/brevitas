# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from numbers import Integral
from typing import Type

import numpy as np
import torch
from torch import nn

from brevitas.common import ExportMixin
from brevitas.common import LayerProtocol
from brevitas.quant_tensor import QuantTensor


@dataclass(frozen=True)
class PWPolyFGeometry:
    num_octaves: int = 5
    exp_bias: int = 127
    exp_base: int = 125
    near_zero_bound: float = 0.25
    clamp_bound: float = 8.0


class PWPolyFEager(nn.Module):

    def __init__(
            self,
            act_impl: Type[nn.Module],
            K,
            degree,
            neg_clamp,
            pos_clamp,
            pos_passthrough,
            fit_samples=1000,
            geometry=None):
        super().__init__()
        self.act_impl = act_impl()
        self.geometry = geometry if geometry is not None else PWPolyFGeometry()
        self.K = self._validate_positive_int(K, "K")
        self.degree = self._validate_positive_int(degree, "degree")
        self.fit_samples = self._validate_positive_int(fit_samples, "fit_samples")
        self.neg_clamp = neg_clamp
        self.pos_clamp = pos_clamp
        self.pos_passthrough = pos_passthrough
        self.num_subs = 1 << self.K
        self.num_segs = 1 + 2 * self.geometry.num_octaves * self.num_subs
        self.register_buffer("coeffs", self.fit_coefficients(), persistent=False)

    def _validate_positive_int(self, value, name):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _segment_boundaries(self):
        bounds = [(-self.geometry.near_zero_bound, self.geometry.near_zero_bound)]

        for octave in range(self.geometry.num_octaves):
            exp_val = self.geometry.exp_base + octave - self.geometry.exp_bias
            base = 2.0 ** exp_val
            for sub in range(self.num_subs):
                lo = base * (1.0 + sub / self.num_subs)
                hi = base * (1.0 + (sub + 1) / self.num_subs)
                bounds.append((lo, hi))

        for octave in range(self.geometry.num_octaves):
            exp_val = self.geometry.exp_base + octave - self.geometry.exp_bias
            base = 2.0 ** exp_val
            for sub in range(self.num_subs):
                lo = base * (1.0 + sub / self.num_subs)
                hi = base * (1.0 + (sub + 1) / self.num_subs)
                bounds.append((-hi, -lo))

        return bounds

    def fit_coefficients(self):
        # fit coefficient tables using the same segmentation as FINN PWPolyF RTL
        bounds = self._segment_boundaries()
        coeffs = np.zeros((len(bounds), self.degree + 1), dtype=np.float64)

        for seg, (lo, hi) in enumerate(bounds):
            xs = np.linspace(lo, hi, self.fit_samples, dtype=np.float64)
            with torch.no_grad():
                ys = self.act_impl(torch.from_numpy(xs).float()).numpy().astype(np.float64)
            coeffs[seg] = np.polynomial.polynomial.polyfit(
                xs, ys, deg=self.degree)[:self.degree + 1]

        return torch.from_numpy(coeffs.astype(np.float32))

    @staticmethod
    def _segment_index(x, K, geometry):
        num_subs = 1 << K
        num_segs = 1 + 2 * geometry.num_octaves * num_subs
        abs_x = x.abs()
        is_neg = x < 0

        is_near_zero = abs_x < geometry.near_zero_bound
        is_clamp = abs_x >= geometry.clamp_bound
        is_neg_clamp = is_neg & is_clamp
        is_pos_clamp = (~is_neg) & is_clamp

        safe_abs = abs_x.clamp(min=geometry.near_zero_bound)
        floor_log2 = torch.floor(torch.log2(safe_abs))
        octave = (floor_log2 + 2).long().clamp(0, geometry.num_octaves - 1)

        pow2 = torch.exp2(floor_log2)
        frac = safe_abs / pow2 - 1.0
        sub = (frac * num_subs).long().clamp(0, num_subs - 1)

        pos_idx = 1 + octave * num_subs + sub
        neg_idx = 1 + geometry.num_octaves * num_subs + octave * num_subs + sub

        seg_idx = torch.where(
            is_near_zero, torch.zeros_like(pos_idx), torch.where(is_neg, neg_idx, pos_idx))
        return seg_idx.clamp(0, num_segs - 1), is_neg_clamp, is_pos_clamp

    @staticmethod
    def evaluate(
            x,
            coeffs,
            K,
            degree,
            neg_clamp,
            pos_clamp,
            pos_passthrough,
            geometry=None):
        geometry = geometry if geometry is not None else PWPolyFGeometry()
        orig_shape = x.shape
        x_flat = x.contiguous().view(-1)

        seg_idx, is_neg_clamp, is_pos_clamp = PWPolyFEager._segment_index(x_flat, K, geometry)
        coeffs = coeffs.to(device=x.device, dtype=x.dtype)
        c = coeffs[seg_idx]

        y = c[:, degree]
        for i in range(degree - 1, -1, -1):
            y = c[:, i] + x_flat * y

        if pos_passthrough:
            pos_val = x_flat
        else:
            pos_val = torch.full_like(y, pos_clamp)
        neg_val = torch.full_like(y, neg_clamp)

        y = torch.where(is_pos_clamp, pos_val, y)
        y = torch.where(is_neg_clamp, neg_val, y)
        return y.view(orig_shape)

    def forward(self, x):
        return self.evaluate(
            x,
            self.coeffs,
            self.K,
            self.degree,
            self.neg_clamp,
            self.pos_clamp,
            self.pos_passthrough,
            self.geometry)


class PWPolyFActivation(nn.Module, ExportMixin, LayerProtocol):

    def __init__(
            self,
            act_impl,
            func,
            neg_clamp,
            pos_clamp,
            pos_passthrough,
            K=3,
            degree=2):
        nn.Module.__init__(self)
        ExportMixin.__init__(self)
        self.eager_impl = PWPolyFEager(
            act_impl,
            K,
            degree,
            neg_clamp=neg_clamp,
            pos_clamp=pos_clamp,
            pos_passthrough=pos_passthrough)
        self.func = func
        self.K = self.eager_impl.K
        self.degree = self.eager_impl.degree

    @property
    def coeffs(self):
        return self.eager_impl.coeffs

    @property
    def requires_export_handler(self):
        return True

    def forward(self, x):
        if isinstance(x, QuantTensor):
            x = x.value
        if x.dtype != torch.float32:
            raise ValueError("FINN PWPolyF requires torch.float32 input")
        if self.export_mode:
            return self.export_handler(x)
        return self.eager_impl(x)


class PWPolyFGELU(PWPolyFActivation):

    def __init__(self, K=3, degree=2):
        PWPolyFActivation.__init__(
            self,
            act_impl=nn.GELU,
            func="gelu",
            neg_clamp=0.0,
            pos_clamp=0.0,
            pos_passthrough=True,
            K=K,
            degree=degree)


class PWPolyFSiLU(PWPolyFActivation):

    def __init__(self, K=3, degree=2):
        PWPolyFActivation.__init__(
            self,
            act_impl=nn.SiLU,
            func="silu",
            neg_clamp=0.0,
            pos_clamp=0.0,
            pos_passthrough=True,
            K=K,
            degree=degree)


class PWPolyFSigmoid(PWPolyFActivation):

    def __init__(self, K=3, degree=2):
        PWPolyFActivation.__init__(
            self,
            act_impl=nn.Sigmoid,
            func="sigmoid",
            neg_clamp=0.0,
            pos_clamp=1.0,
            pos_passthrough=False,
            K=K,
            degree=degree)


class PWPolyFTanh(PWPolyFActivation):

    def __init__(self, K=3, degree=2):
        PWPolyFActivation.__init__(
            self,
            act_impl=nn.Tanh,
            func="tanh",
            neg_clamp=-1.0,
            pos_clamp=1.0,
            pos_passthrough=False,
            K=K,
            degree=degree)
