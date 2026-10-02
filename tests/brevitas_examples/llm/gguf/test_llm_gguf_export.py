# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from gguf import QK_K
import gguf.quants as gguf_quants
import numpy as np
import pytest
import pytest_cases

from brevitas_examples.llm.gguf_export.convert import SUPPORTED_OVERRIDE_QTYPES
from brevitas_examples.llm.gguf_export.quant import q2_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q3_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_0_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_1_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q5_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q6_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q8_0_quant_block

from .common import *

# --- 32-element block packers: Q4_0, Q4_1, Q8_0 ---


@pytest.mark.llm
def test_q4_0_pack():
    """Lossless pack: signed 4-bit codes + fp16 scale d decode to exactly code * d."""
    rng = np.random.default_rng(0)
    codes = rng.integers(-8, 8, size=(8, 32)).astype(np.float32)  # signed [-8, 7]
    d = (np.abs(rng.standard_normal((8, 1))) + 0.1).astype(np.float32)
    x_hat = gguf_quants.dequantize(q4_0_quant_block(codes.copy(), scale=d), Q4_0)
    np.testing.assert_allclose(x_hat, codes * fp16(d), rtol=0, atol=0)


@pytest.mark.llm
def test_q4_1_pack():
    """Lossless pack: codes + scale d and zero-point zp decode to exactly code * d + min."""
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 16, size=(8, 32)).astype(np.float32)  # unsigned [0, 15]
    d = (np.abs(rng.standard_normal((8, 1))) + 0.1).astype(np.float32)
    zp = rng.standard_normal((8, 1)).astype(np.float32)
    m = -zp * d  # q4_1 stores min = -zp * d
    x_hat = gguf_quants.dequantize(q4_1_quant_block(codes.copy(), scale=d, zp=zp), Q4_1)
    np.testing.assert_allclose(x_hat, codes * fp16(d) + fp16(m), rtol=0, atol=0)


@pytest.mark.llm
def test_q8_0_pack():
    """Lossless pack: int8 codes + fp16 scale d decode to exactly code * d."""
    rng = np.random.default_rng(1)
    codes = rng.integers(-127, 128, size=(8, 32)).astype(np.float32)
    d = (np.abs(rng.standard_normal((8, 1))) + 0.1).astype(np.float32)
    x_hat = gguf_quants.dequantize(q8_0_quant_block(codes.copy(), scale=d), Q8_0)
    np.testing.assert_allclose(x_hat, codes * fp16(d), rtol=0, atol=0)


# --- K-quant super-block packers: Q2_K through Q6_K ---


@pytest.mark.llm
def test_q2_k_pack():
    """Lossless pack: 2-bit codes + sub-block scales/mins and fp16 super-scales decode
    to exactly d_scale*qs*code - d_wmin*qm. Q2_K mirrors Q4_K with 16 sub-blocks."""
    rng = np.random.default_rng(5)
    nb = 4
    n_sub = QK_K // 16
    codes = rng.integers(0, 4, size=(nb, QK_K)).astype(np.float32)
    scales = (np.abs(rng.standard_normal((nb, n_sub))) + 0.1).astype(np.float32)
    mins = (np.abs(rng.standard_normal((nb, n_sub))) + 0.1).astype(np.float32)
    d_scale = scales.max(1, keepdims=True) / 15
    d_wmin = mins.max(1, keepdims=True) / 15
    q = q2_k_quant_block(codes.copy(), scale=scales, wmin_m=mins, d_scale=d_scale, d_wmin_m=d_wmin)
    x_hat = gguf_quants.dequantize(q, Q2_K).reshape(nb, n_sub, 16)
    qs = np.round(scales / d_scale).clip(0, 15)
    qm = np.round(mins / d_wmin).clip(0, 15)
    expected = (
        fp16(d_scale)[:, :, None] * qs[:, :, None] * codes.reshape(nb, n_sub, 16) -
        fp16(d_wmin)[:, :, None] * qm[:, :, None])
    np.testing.assert_allclose(x_hat, expected, rtol=0, atol=0)


@pytest.mark.llm
def test_q3_k_pack():
    """Lossless pack: signed 3-bit codes + sub-block scales (6-bit, fp16 super-d) decode
    to exactly (d*q_scale)*code."""
    rng = np.random.default_rng(6)
    nb = 4
    n_sub = QK_K // 16
    codes = rng.integers(-4, 4, size=(nb, QK_K)).astype(np.float32)
    scales = (np.abs(rng.standard_normal((nb, n_sub))) + 0.05).astype(np.float32)
    d_scale = scales.max(1, keepdims=True) / 32
    q = q3_k_quant_block(codes.copy(), scale=scales, d_scale=d_scale)
    x_hat = gguf_quants.dequantize(q, Q3_K).reshape(nb, n_sub, 16)
    q_scales = np.round(scales / d_scale).clip(-32, 31)
    eff = fp16(d_scale)[:, :, None] * q_scales[:, :, None]
    expected = eff * codes.reshape(nb, n_sub, 16)
    np.testing.assert_allclose(x_hat, expected, rtol=0, atol=0)


@pytest.mark.llm
def test_q4_k_pack():
    """Lossless pack: codes + sub-block scales/mins and fp16 super-scales decode to
    exactly d_scale*qs*code - d_wmin*qm."""
    rng = np.random.default_rng(3)
    nb = 4
    codes = rng.integers(0, 16, size=(nb, QK_K)).astype(np.float32)
    scales = (np.abs(rng.standard_normal((nb, 8))) + 0.1).astype(np.float32)
    mins = (np.abs(rng.standard_normal((nb, 8))) + 0.1).astype(np.float32)
    d_scale = scales.max(1, keepdims=True) / 63
    d_wmin = mins.max(1, keepdims=True) / 63
    q = q4_k_quant_block(codes.copy(), scale=scales, wmin_m=mins, d_scale=d_scale, d_wmin_m=d_wmin)
    x_hat = gguf_quants.dequantize(q, Q4_K).reshape(nb, 8, 32)
    qs = np.round(scales / d_scale).clip(0, 63)
    qm = np.round(mins / d_wmin).clip(0, 63)
    expected = (
        fp16(d_scale)[:, :, None] * qs[:, :, None] * codes.reshape(nb, 8, 32) -
        fp16(d_wmin)[:, :, None] * qm[:, :, None])
    np.testing.assert_allclose(x_hat, expected, rtol=0, atol=0)


@pytest.mark.llm
def test_q5_k_pack():
    """Lossless pack: 5-bit codes + sub-block scales/mins and fp16 super-scales decode
    to exactly d_scale*qs*code - d_wmin*qm. Q5_K mirrors Q4_K with codes in [0, 31]."""
    rng = np.random.default_rng(3)
    nb = 4
    codes = rng.integers(0, 32, size=(nb, QK_K)).astype(np.float32)  # 5-bit [0, 31]
    scales = (np.abs(rng.standard_normal((nb, 8))) + 0.1).astype(np.float32)
    mins = (np.abs(rng.standard_normal((nb, 8))) + 0.1).astype(np.float32)
    d_scale = scales.max(1, keepdims=True) / 63
    d_wmin = mins.max(1, keepdims=True) / 63
    q = q5_k_quant_block(codes.copy(), scale=scales, wmin_m=mins, d_scale=d_scale, d_wmin_m=d_wmin)
    x_hat = gguf_quants.dequantize(q, Q5_K).reshape(nb, 8, 32)
    qs = np.round(scales / d_scale).clip(0, 63)
    qm = np.round(mins / d_wmin).clip(0, 63)
    expected = (
        fp16(d_scale)[:, :, None] * qs[:, :, None] * codes.reshape(nb, 8, 32) -
        fp16(d_wmin)[:, :, None] * qm[:, :, None])
    np.testing.assert_allclose(x_hat, expected, rtol=0, atol=0)


@pytest.mark.llm
def test_q6_k_pack():
    """Lossless pack: signed 6-bit codes + sub-block scales (int8, fp16 super-d) decode
    to exactly (d*q_scale)*code."""
    rng = np.random.default_rng(4)
    nb = 4
    n_sub = QK_K // 16
    codes = rng.integers(-32, 32, size=(nb, QK_K)).astype(np.float32)  # signed [-32, 31]
    scales = (np.abs(rng.standard_normal((nb, n_sub))) + 0.05).astype(np.float32)
    d_scale = scales.max(1, keepdims=True) / 128
    q = q6_k_quant_block(codes.copy(), scale=scales, d_scale=d_scale)
    x_hat = gguf_quants.dequantize(q, Q6_K).reshape(nb, n_sub, 16)
    q_scales = np.round(scales / d_scale).clip(-128, 127)
    eff = fp16(d_scale)[:, :, None] * q_scales[:, :, None]
    expected = eff * codes.reshape(nb, n_sub, 16)
    np.testing.assert_allclose(x_hat, expected, rtol=0, atol=0)


# --- Override qtype round-trip through gguf.quants.quantize ---


@pytest.mark.llm
@pytest_cases.parametrize(
    "qtype", list(SUPPORTED_OVERRIDE_QTYPES), ids=[t.name for t in SUPPORTED_OVERRIDE_QTYPES])
def test_override_qtype_encodes(qtype):
    # Every override qtype must round-trip through gguf.quants.quantize -- via a native
    # encoder (Q4_0/Q4_1/Q8_0) or a float cast (F32/F16). Guards the registry
    # ModelBase asserts against.
    x = normal(0, 8)
    x_hat = gguf_quants.dequantize(gguf_quants.quantize(x, qtype), qtype).reshape(x.shape)
    assert np.isfinite(x_hat).all()
