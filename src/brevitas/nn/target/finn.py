# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

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


@dataclass(frozen=True)
class PWPolyFFunctionSpec:
    name: str
    reference_impl: Callable[[torch.Tensor], torch.Tensor]
    neg_clamp: float
    pos_clamp: float
    pos_passthrough: bool


class PWPolyFEager:
    def __init__(self, func, K, degree, fit_samples=1000, geometry=None, function_specs=None):
        self.geometry = geometry if geometry is not None else PWPolyFGeometry()
        self.function_specs = (
            function_specs if function_specs is not None else self.default_function_specs())
        self.func = self._validate_func(func)
        self.K = self._validate_positive_int(K, "K")
        self.degree = self._validate_positive_int(degree, "degree")
        self.fit_samples = self._validate_positive_int(fit_samples, "fit_samples")
        self.function_spec = self.function_specs[self.func]
        self.num_subs = 1 << self.K
        self.num_segs = 1 + 2 * self.geometry.num_octaves * self.num_subs

    @classmethod
    def default_function_specs(cls):
        specs = (
            PWPolyFFunctionSpec("gelu", F.gelu, 0.0, 0.0, True),
            PWPolyFFunctionSpec("silu", F.silu, 0.0, 0.0, True),
            PWPolyFFunctionSpec("sigmoid", torch.sigmoid, 0.0, 1.0, False),
            PWPolyFFunctionSpec("tanh", torch.tanh, -1.0, 1.0, False),)
        return {spec.name: spec for spec in specs}

    @classmethod
    def supported_funcs(cls):
        return tuple(cls.default_function_specs().keys())

    def _validate_positive_int(self, value, name):
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def _validate_func(self, func):
        if func not in self.function_specs:
            raise ValueError(
                "Unsupported func=%r; choose from %s" % (func, tuple(self.function_specs)))
        return func

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
                ys = self.function_spec.reference_impl(
                    torch.from_numpy(xs).float()).numpy().astype(np.float64)
            coeffs[seg] = np.polynomial.polynomial.polyfit(
                xs, ys, deg=self.degree)[:self.degree + 1]

        return torch.from_numpy(coeffs.astype(np.float32))

    def _segment_index(self, x):
        abs_x = x.abs()
        is_neg = x < 0

        is_near_zero = abs_x < self.geometry.near_zero_bound
        is_clamp = abs_x >= self.geometry.clamp_bound
        is_neg_clamp = is_neg & is_clamp
        is_pos_clamp = (~is_neg) & is_clamp

        safe_abs = abs_x.clamp(min=self.geometry.near_zero_bound)
        floor_log2 = torch.floor(torch.log2(safe_abs))
        octave = (floor_log2 + 2).long().clamp(0, self.geometry.num_octaves - 1)

        pow2 = torch.exp2(floor_log2)
        frac = safe_abs / pow2 - 1.0
        sub = (frac * self.num_subs).long().clamp(0, self.num_subs - 1)

        pos_idx = 1 + octave * self.num_subs + sub
        neg_idx = (
            1 + self.geometry.num_octaves * self.num_subs + octave * self.num_subs + sub)

        seg_idx = torch.where(
            is_near_zero, torch.zeros_like(pos_idx), torch.where(is_neg, neg_idx, pos_idx))
        return seg_idx.clamp(0, self.num_segs - 1), is_neg_clamp, is_pos_clamp

    def evaluate(self, x, coeffs):
        orig_shape = x.shape
        x_flat = x.contiguous().view(-1)

        seg_idx, is_neg_clamp, is_pos_clamp = self._segment_index(x_flat)
        coeffs = coeffs.to(device=x.device, dtype=x.dtype)
        c = coeffs[seg_idx]

        y = c[:, self.degree]
        for i in range(self.degree - 1, -1, -1):
            y = c[:, i] + x_flat * y

        if self.function_spec.pos_passthrough:
            pos_val = x_flat
        else:
            pos_val = torch.full_like(y, self.function_spec.pos_clamp)
        neg_val = torch.full_like(y, self.function_spec.neg_clamp)

        y = torch.where(is_pos_clamp, pos_val, y)
        y = torch.where(is_neg_clamp, neg_val, y)
        return y.view(orig_shape)


class PWPolyFActivation(nn.Module, ExportMixin, LayerProtocol):
    def __init__(self, func="gelu", K=3, degree=2):
        nn.Module.__init__(self)
        ExportMixin.__init__(self)
        self.eager_impl = PWPolyFEager(func, K, degree)
        self.func = self.eager_impl.func
        self.K = self.eager_impl.K
        self.degree = self.eager_impl.degree
        coeffs = self.eager_impl.fit_coefficients()
        self.register_buffer("coeffs", coeffs, persistent=False)

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
        return self.eager_impl.evaluate(x, self.coeffs)
