# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import pytest_cases

pytest.importorskip("gguf")

import gguf
import gguf.quants as gguf_quants

from brevitas_examples.llm.gguf_export.quant import _q6_k_quantize_scales
from brevitas_examples.llm.gguf_export.quant import GGML_QUANT_SIZES
from brevitas_examples.llm.gguf_export.quant import q4_0_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_1_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q6_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q8_0_quant_block
from brevitas_examples.llm.gguf_export.quant import QK_K

Q4_0 = gguf.GGMLQuantizationType.Q4_0
Q4_1 = gguf.GGMLQuantizationType.Q4_1
Q8_0 = gguf.GGMLQuantizationType.Q8_0
Q4_K = gguf.GGMLQuantizationType.Q4_K
Q6_K = gguf.GGMLQuantizationType.Q6_K
# block_q6_K (ggml-common.h): ql[QK_K/2] + qh[QK_K/4] + scales[QK_K/16] + d(fp16)
Q6_K_TYPE_SIZE = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2


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

# (encoder, qtype) for the quants that gguf ships
NATIVE_GGUF_QUANTIZERS = {
    "q4_0": (q4_0_quant_block, Q4_0),
    "q4_1": (q4_1_quant_block, Q4_1),
    "q8_0": (q8_0_quant_block, Q8_0),}


@pytest.mark.llm
def test_q6_k_block_size_matches_format():
    # The numpy adaptation must agree with the on-disk block_q6_K size, and the
    # encoder must emit exactly that many bytes per 256-element block.
    assert GGML_QUANT_SIZES[Q6_K] == (QK_K, Q6_K_TYPE_SIZE)
    q = q6_k_quant_block(_normal(0, 5))
    assert q.dtype == np.uint8
    assert q.shape == (5, Q6_K_TYPE_SIZE)


@pytest.mark.llm
@pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
def test_q6_k_quant_error(x):
    # Quantize then decode; this test verifies that every element is within one Q6_K
    # step. The 64 signed levels span [-amax, amax], so the step is amax/32 and the
    # round-to-nearest floor is amax/64; one extra full step (2x) absorbs possible
    # quantization error from the 8-bit scale, the fp16 super-block scale, and the
    # scale search.
    x_hat = gguf_quants.dequantize(q6_k_quant_block(x), Q6_K)
    amax = np.abs(x).max()
    assert np.abs(x - x_hat).max() <= amax / 32


@pytest.mark.llm
def test_q6_k_gguf_quantize_dispatch():
    # Importing the export module monkey-patches gguf.quants.Q6_K.quantize_blocks
    # so gguf.quants.quantize(data, Q6_K) routes through our encoder. Without
    # this, the convert.py fallback would silently regress Q6_K targets to F32.
    x = _normal(5, 4)
    via_gguf = gguf_quants.quantize(x, Q6_K)
    np.testing.assert_array_equal(via_gguf.reshape(-1, Q6_K_TYPE_SIZE), q6_k_quant_block(x))


@pytest.mark.llm
@pytest_cases.parametrize(
    "fn,qtype", list(NATIVE_GGUF_QUANTIZERS.values()), ids=list(NATIVE_GGUF_QUANTIZERS))
def test_self_quantize_matches_native(fn, qtype):
    # If scale=None, then our quantization path must match the native gguf quantization
    # byte-for-byte. This is done at export (e.g., override_model_tensors)
    x = _normal(0, 64, block=32)
    ours = fn(x.copy(), scale=None)
    native = gguf_quants.quantize(x, qtype).reshape(ours.shape)
    np.testing.assert_array_equal(ours, native)


@pytest.mark.llm
@pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
def test_q4_k_quant_error(x):
    # Quantize then decode natively; this tests that every element is within one Q4_K
    # step. Each 32-element sub-block maps its own [min, max] onto 16 levels, so the
    # sub-block holding the global extreme spans at most 2*amax and its step is at most
    # 2*amax/15; one extra step (2x the amax/15 round-to-nearest floor) absorbs the
    # 6-bit scale/min quantization and the fp16 super-block scales.
    x_hat = gguf_quants.dequantize(q4_k_quant_block(x), Q4_K)
    amax = np.abs(x).max()
    assert np.abs(x - x_hat).max() <= 2 * amax / 15


@pytest.mark.llm
def test_q4_0_pack():
    # Pack mode: blocks are pre-quantized signed 4-bit codes and scale is the fp16
    # per-block scale d. Packing only lays out bytes, so it is lossless: a native
    # decode must return exactly code * d (hence atol=0).
    rng = np.random.default_rng(0)
    codes = rng.integers(-8, 8, size=(8, 32)).astype(np.float32)  # signed [-8, 7]
    d = (np.abs(rng.standard_normal((8, 1))) + 0.1).astype(np.float32)
    x_hat = gguf_quants.dequantize(q4_0_quant_block(codes.copy(), scale=d), Q4_0)
    np.testing.assert_allclose(x_hat, codes * fp16(d), rtol=0, atol=0)


@pytest.mark.llm
def test_q8_0_pack():
    # Pack mode: pre-quantized int8 codes plus the fp16 per-block scale d. Lossless
    # byte layout, so a native decode must return exactly code * d.
    rng = np.random.default_rng(1)
    codes = rng.integers(-127, 128, size=(8, 32)).astype(np.float32)
    d = (np.abs(rng.standard_normal((8, 1))) + 0.1).astype(np.float32)
    x_hat = gguf_quants.dequantize(q8_0_quant_block(codes.copy(), scale=d), Q8_0)
    np.testing.assert_allclose(x_hat, codes * fp16(d), rtol=0, atol=0)


@pytest.mark.llm
def test_q4_1_pack():
    # Pack mode for the asymmetric quant: pre-quantized unsigned 4-bit codes, with
    # scale d and zero-point zp, where q4_1 stores the offset as min = -zp * d.
    # Lossless byte layout, so a native decode must return exactly code * d + min.
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 16, size=(8, 32)).astype(np.float32)  # unsigned [0, 15]
    d = (np.abs(rng.standard_normal((8, 1))) + 0.1).astype(np.float32)
    zp = rng.standard_normal((8, 1)).astype(np.float32)
    m = -zp * d  # q4_1 stores min = -zp * d
    x_hat = gguf_quants.dequantize(q4_1_quant_block(codes.copy(), scale=d, zp=zp), Q4_1)
    np.testing.assert_allclose(x_hat, codes * fp16(d) + fp16(m), rtol=0, atol=0)


@pytest.mark.llm
def test_q4_k_pack():
    # Pack mode for the two-level K-quant: pre-quantized 4-bit codes plus the 8
    # sub-block scales/mins and their fp16 super-scales (d_scale, d_wmin). Packing
    # 6-bit-quantizes the sub-block scales/mins against the super-scales, so a native
    # decode returns exactly d_scale*qs*code - d_wmin*qm; assert that reconstruction.
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
    # Pack mode: pre-quantized signed 6-bit codes plus the 16 per-sub-block scales.
    # Packing quantizes those scales to int8 codes against an fp16 super-block scale
    # d, so a native decode returns exactly (d * q_scale) * code; assert that.
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
