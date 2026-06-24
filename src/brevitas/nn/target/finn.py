# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from brevitas.common import ExportMixin
from brevitas.common import LayerProtocol
from brevitas.quant_tensor import QuantTensor

__all__ = [
    "PWPolyFActivation",
    "SUPPORTED_PWPOLYF_FUNCS",
    "fit_pwpolyf_coefficients",
    "pwpolyf_eager",
    "pwpolyf_eager_from_attrs",
]

NUM_OCTAVES = 5
EXP_BIAS = 127
EXP_BASE = 125
DEFAULT_K = 3
DEFAULT_DEGREE = 2
DEFAULT_FIT_SAMPLES = 1000

SUPPORTED_PWPOLYF_FUNCS = ("gelu", "silu", "sigmoid", "tanh")

_REFERENCE_FUNCS = {
    "gelu": F.gelu,
    "silu": F.silu,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,}

_CLAMP_CFG = {
    "gelu": {
        "neg_clamp": 0.0,
        "pos_clamp": 0.0,
        "pos_passthrough": True,},
    "silu": {
        "neg_clamp": 0.0,
        "pos_clamp": 0.0,
        "pos_passthrough": True,},
    "sigmoid": {
        "neg_clamp": 0.0,
        "pos_clamp": 1.0,
        "pos_passthrough": False,},
    "tanh": {
        "neg_clamp": -1.0,
        "pos_clamp": 1.0,
        "pos_passthrough": False,},}


def _validate_positive_int(value, name):
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_func(func):
    if func not in SUPPORTED_PWPOLYF_FUNCS:
        raise ValueError(
            "Unsupported func=%r; choose from %s" % (func, SUPPORTED_PWPOLYF_FUNCS))
    return func


def _segment_boundaries(K):
    num_subs = 1 << K
    bounds = [(-0.25, 0.25)]

    for octave in range(NUM_OCTAVES):
        exp_val = EXP_BASE + octave - EXP_BIAS
        base = 2.0 ** exp_val
        for sub in range(num_subs):
            lo = base * (1.0 + sub / num_subs)
            hi = base * (1.0 + (sub + 1) / num_subs)
            bounds.append((lo, hi))

    for octave in range(NUM_OCTAVES):
        exp_val = EXP_BASE + octave - EXP_BIAS
        base = 2.0 ** exp_val
        for sub in range(num_subs):
            lo = base * (1.0 + sub / num_subs)
            hi = base * (1.0 + (sub + 1) / num_subs)
            bounds.append((-hi, -lo))

    return bounds


def fit_pwpolyf_coefficients(func, K, degree, num_samples=DEFAULT_FIT_SAMPLES):
    """Fit coefficient tables using the same segmentation as FINN PWPolyF RTL."""
    func = _validate_func(func)
    K = _validate_positive_int(K, "K")
    degree = _validate_positive_int(degree, "degree")
    num_samples = _validate_positive_int(num_samples, "num_samples")

    ref_func = _REFERENCE_FUNCS[func]
    bounds = _segment_boundaries(K)
    coeffs = np.zeros((len(bounds), degree + 1), dtype=np.float64)

    for seg, (lo, hi) in enumerate(bounds):
        xs = np.linspace(lo, hi, num_samples, dtype=np.float64)
        with torch.no_grad():
            ys = ref_func(torch.from_numpy(xs).float()).numpy().astype(np.float64)
        coeffs[seg] = np.polynomial.polynomial.polyfit(xs, ys, deg=degree)[:degree + 1]

    return torch.from_numpy(coeffs.astype(np.float32))


def _segment_index(x, K, num_subs, num_segs):
    abs_x = x.abs()
    is_neg = x < 0

    is_near_zero = abs_x < 0.25
    is_clamp = abs_x >= 8.0
    is_neg_clamp = is_neg & is_clamp
    is_pos_clamp = (~is_neg) & is_clamp

    safe_abs = abs_x.clamp(min=0.25)
    floor_log2 = torch.floor(torch.log2(safe_abs))
    octave = (floor_log2 + 2).long().clamp(0, NUM_OCTAVES - 1)

    pow2 = torch.exp2(floor_log2)
    frac = safe_abs / pow2 - 1.0
    sub = (frac * num_subs).long().clamp(0, num_subs - 1)

    pos_idx = 1 + octave * num_subs + sub
    neg_idx = 1 + NUM_OCTAVES * num_subs + octave * num_subs + sub

    seg_idx = torch.where(
        is_near_zero, torch.zeros_like(pos_idx), torch.where(is_neg, neg_idx, pos_idx))
    return seg_idx.clamp(0, num_segs - 1), is_neg_clamp, is_pos_clamp


def pwpolyf_eager(x, coeffs, func, K, degree):
    func = _validate_func(func)
    K = _validate_positive_int(K, "K")
    degree = _validate_positive_int(degree, "degree")

    num_subs = 1 << K
    num_segs = 1 + 2 * NUM_OCTAVES * num_subs
    orig_shape = x.shape
    x_flat = x.contiguous().view(-1)

    seg_idx, is_neg_clamp, is_pos_clamp = _segment_index(x_flat, K, num_subs, num_segs)
    coeffs = coeffs.to(device=x.device, dtype=x.dtype)
    c = coeffs[seg_idx]

    y = c[:, degree]
    for i in range(degree - 1, -1, -1):
        y = c[:, i] + x_flat * y

    cfg = _CLAMP_CFG[func]
    if cfg["pos_passthrough"]:
        pos_val = x_flat
    else:
        pos_val = torch.full_like(y, cfg["pos_clamp"])
    neg_val = torch.full_like(y, cfg["neg_clamp"])

    y = torch.where(is_pos_clamp, pos_val, y)
    y = torch.where(is_neg_clamp, neg_val, y)
    return y.view(orig_shape)


def pwpolyf_eager_from_attrs(x, func, K, degree):
    coeffs = fit_pwpolyf_coefficients(func, K, degree)
    return pwpolyf_eager(x, coeffs, func, K, degree)


# PWPolyF activation approximation for FINN
class PWPolyFActivation(nn.Module, ExportMixin, LayerProtocol):
    def __init__(self, func="gelu", K=DEFAULT_K, degree=DEFAULT_DEGREE):
        nn.Module.__init__(self)
        ExportMixin.__init__(self)
        self.func = _validate_func(func)
        self.K = _validate_positive_int(K, "K")
        self.degree = _validate_positive_int(degree, "degree")
        coeffs = fit_pwpolyf_coefficients(self.func, self.K, self.degree)
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
        return pwpolyf_eager(x, self.coeffs, self.func, self.K, self.degree)
