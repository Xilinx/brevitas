# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import gguf
from gguf import GGML_QUANT_SIZES
from gguf import QK_K
import gguf.quants as gguf_quants
import numpy as np
import pytest
import pytest_cases

from brevitas_examples.llm.gguf_export.convert import SUPPORTED_OVERRIDE_QTYPES
from brevitas_examples.llm.gguf_export.quant import _q6_k_quantize_scales
from brevitas_examples.llm.gguf_export.quant import q4_0_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_1_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q6_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q8_0_quant_block

Q4_0 = gguf.GGMLQuantizationType.Q4_0
Q4_1 = gguf.GGMLQuantizationType.Q4_1
Q8_0 = gguf.GGMLQuantizationType.Q8_0
Q4_K = gguf.GGMLQuantizationType.Q4_K
Q6_K = gguf.GGMLQuantizationType.Q6_K


def fp16(a):
    """Round-trip through fp16, matching how scales are stored on disk."""
    return a.astype(np.float16).astype(np.float32)


def _normal(seed: int, nb: int, block: int = QK_K):
    return np.random.default_rng(seed).standard_normal((nb, block)).astype(np.float32)


def _outlier():
    # A large spike forces a wide sub-block scale and the [-32, 31] clamp.
    x = _normal(9, 4)
    x[:, 0] = 50.0
    return x


# Random model tensors varying block counts, constants (incl. negative),
# zero, and a high-dynamic-range spike.
MODEL_TENSORS = {
    "normal_1blk": _normal(0, 1),
    "normal_4blk": _normal(1, 4),
    "normal_17blk": _normal(7, 17),
    "const_pos": np.full((2, QK_K), 0.37, dtype=np.float32),
    "const_large": np.full((2, QK_K), 5.0, dtype=np.float32),
    "const_neg": np.full((2, QK_K), -2.3, dtype=np.float32),
    "zero": np.zeros((2, QK_K), dtype=np.float32),
    "outlier": _outlier(),}


@pytest.mark.llm
class TestQ6KQuant:
    """gguf ships no Q6_K quantizer, so the export module monkey-patches
    gguf.quants.Q6_K.quantize_blocks with our numpy encoder. These tests cover the
    dispatch wiring and the encoder's block size and accuracy."""

    encoder = staticmethod(q6_k_quant_block)
    qtype = Q6_K

    def test_dispatch(self):
        """gguf.quants.quantize(data, Q6_K) routes through our patched encoder.

        Without the patch the convert.py pass-through path would regress Q6_K
        targets to F32."""
        x = _normal(5, 4)
        via_gguf = gguf_quants.quantize(x, self.qtype)
        type_size = GGML_QUANT_SIZES[self.qtype][1]
        np.testing.assert_array_equal(via_gguf.reshape(-1, type_size), self.encoder(x))

    def test_block_size(self):
        """The encoder emits exactly the on-disk block size from GGML_QUANT_SIZES."""
        _, type_size = GGML_QUANT_SIZES[self.qtype]
        q = self.encoder(_normal(0, 5))
        assert q.dtype == np.uint8
        assert q.shape == (5, type_size)

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_quant_error(self, x):
        """Quantize then decode; every element lands within one Q6_K step.

        The 64 signed levels span [-amax, amax], so the step is roughly amax/2^5
        and the round-to-nearest floor is s / 2; we bound by one extra full step (2x)
        to account for the possible deviation induced from scale search or error from
        scale quantization, which gives us s = amax / 32"""
        x_hat = gguf_quants.dequantize(self.encoder(x), self.qtype)
        amax = np.abs(x).max()
        assert np.abs(x - x_hat).max() <= amax / 32


@pytest.mark.llm
@pytest_cases.parametrize(
    "qtype", list(SUPPORTED_OVERRIDE_QTYPES), ids=[t.name for t in SUPPORTED_OVERRIDE_QTYPES])
def test_override_qtype_encodes(qtype):
    # Every override qtype must round-trip through gguf.quants.quantize -- via a native
    # encoder (Q4_0/Q4_1/Q8_0), a float cast (F32/F16), or one of our monkey-patched
    # K-quant encoders (Q4_K/Q6_K). Guards the registry ModelBase asserts against.
    x = _normal(0, 8)
    x_hat = gguf_quants.dequantize(gguf_quants.quantize(x, qtype), qtype).reshape(x.shape)
    assert np.isfinite(x_hat).all()


@pytest.mark.llm
def test_q4_0_pack():
    """Lossless pack: signed 4-bit codes + fp16 scale d decode to exactly code * d."""
    rng = np.random.default_rng(0)
    codes = rng.integers(-8, 8, size=(8, 32)).astype(np.float32)  # signed [-8, 7]
    d = (np.abs(rng.standard_normal((8, 1))) + 0.1).astype(np.float32)
    x_hat = gguf_quants.dequantize(q4_0_quant_block(codes.copy(), scale=d), Q4_0)
    np.testing.assert_allclose(x_hat, codes * fp16(d), rtol=0, atol=0)


@pytest.mark.llm
def test_q8_0_pack():
    """Lossless pack: int8 codes + fp16 scale d decode to exactly code * d."""
    rng = np.random.default_rng(1)
    codes = rng.integers(-127, 128, size=(8, 32)).astype(np.float32)
    d = (np.abs(rng.standard_normal((8, 1))) + 0.1).astype(np.float32)
    x_hat = gguf_quants.dequantize(q8_0_quant_block(codes.copy(), scale=d), Q8_0)
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
def test_q6_k_pack():
    """Lossless pack: signed 6-bit codes + sub-block scales (int8, fp16 super-d) decode
    to exactly (d*q_scale)*code."""
    rng = np.random.default_rng(4)
    nb = 4
    codes = rng.integers(-32, 32, size=(nb, QK_K)).astype(np.float32)  # signed [-32, 31]
    scales = (np.abs(rng.standard_normal((nb, QK_K // 16))) + 0.05).astype(np.float32)
    q = q6_k_quant_block(codes.copy(), scale=scales)
    x_hat = gguf_quants.dequantize(q, Q6_K).reshape(nb, QK_K // 16, 16)
    d, q_scales, _ = _q6_k_quantize_scales(scales)
    eff = fp16(d)[:, None] * q_scales.astype(np.float32)  # effective per-sub-block scale
    expected = eff[:, :, None] * codes.reshape(nb, QK_K // 16, 16)
    np.testing.assert_allclose(x_hat, expected, rtol=0, atol=0)
