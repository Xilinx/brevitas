# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

pytest.importorskip("gguf")

import gguf
import gguf.quants as gguf_quants

from brevitas_examples.llm.gguf_export.quant import GGML_QUANT_SIZES
from brevitas_examples.llm.gguf_export.quant import q6_k_quant_block
from brevitas_examples.llm.gguf_export.quant import QK_K

Q6_K = gguf.GGMLQuantizationType.Q6_K
# block_q6_K (ggml-common.h): ql[QK_K/2] + qh[QK_K/4] + scales[QK_K/16] + d(fp16)
Q6_K_TYPE_SIZE = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2


def _dequantize(q_bytes: np.ndarray, qtype) -> np.ndarray:
    """Decode packed blocks with gguf's native dequantizer."""
    return gguf_quants.dequantize(q_bytes.reshape(-1), qtype)


def _normal(seed: int, nb: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((nb, QK_K)).astype(np.float32)


def _outlier() -> np.ndarray:
    # A large spike forces a wide sub-block scale and the [-32, 31] clamp.
    x = _normal(9, 4)
    x[:, 0] = 50.0
    return x


@pytest.mark.llm
def test_q6_k_block_size_matches_format():
    # The numpy port must agree with the on-disk block_q6_K size, and the
    # encoder must emit exactly that many bytes per 256-element block.
    assert GGML_QUANT_SIZES[Q6_K] == (QK_K, Q6_K_TYPE_SIZE)
    q = q6_k_quant_block(_normal(0, 5))
    assert q.dtype == np.uint8
    assert q.shape == (5, Q6_K_TYPE_SIZE)


TEST_MODEL_TENSORS = {
    "normal_1blk": _normal(0, 1),
    "normal_4blk": _normal(1, 4),
    "normal_17blk": _normal(7, 17),
    "const_pos": np.full((2, QK_K), 0.37, dtype=np.float32),
    "const_large": np.full((2, QK_K), 5.0, dtype=np.float32),
    "const_neg": np.full((2, QK_K), -2.3, dtype=np.float32),
    "zero": np.zeros((2, QK_K), dtype=np.float32),
    "outlier": _outlier(),}


@pytest.mark.llm
@pytest.mark.parametrize("x", TEST_MODEL_TENSORS.values(), ids=list(TEST_MODEL_TENSORS))
def test_q6_k_roundtrip_linf(x):
    # Encode (quantize + pack) with our code, decode with gguf's native decoder,
    # and require every element to land within one Q6_K step. The 64 signed levels
    # span [-amax, amax], so the step is amax/32 and the round-to-nearest floor
    # is amax/64; we allow one full step of error (2x) to absorb the 8-bit
    # scale-code quantization, the fp16 super-block scale, and the scale search.
    x_hat = _dequantize(q6_k_quant_block(x), Q6_K).reshape(x.shape)
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
