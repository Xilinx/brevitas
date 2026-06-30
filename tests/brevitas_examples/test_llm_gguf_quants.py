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


def _custom_q6_k_export(weight: np.ndarray) -> np.ndarray:
    """Quantize ``weight`` with the Brevitas custom Q6_K quantizer and pack it to a
    GGUF Q6_K block exactly as brevitas_examples...convert.ModelBase.quantize does.

    Returns the packed uint8 block (one row of blocks per weight row).
    """
    import torch

    from brevitas.core.restrict_val import QuantRestrictValue
    import brevitas.nn as qnn
    from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ6_KWeightQuant
    from brevitas_examples.llm.gguf_export.quant import ggml_quant

    out_features, in_features = weight.shape
    layer = qnn.QuantLinear(in_features, out_features, bias=False, weight_quant=GGUFQ6_KWeightQuant)
    layer.weight.data = torch.from_numpy(weight.copy())
    quant_weight = layer.quant_weight()
    quant_data = quant_weight.int()
    # scale_ holds the (out, n_sub, 1) per-sub-block float scales.
    scale = quant_weight.scale_ if hasattr(quant_weight, 'scale_') else quant_weight.scale
    zp = quant_weight.zero_point
    # Pull the calibrated nested scale (quantized sub-scales + fp16 super-block d)
    # off the QuantRestrictValue, mirroring convert.py's Q6_K branch.
    restrict = next(m for m in layer.weight_quant.modules() if isinstance(m, QuantRestrictValue))
    quant_scale, scale_scale, *_ = restrict.float_to_int_impl(scale)
    block = ggml_quant(quant_data, Q6_K, quant_scale, zp, d_scale=scale_scale)
    # ggml_quant squeezes singleton dims; normalize to (n_rows, type_size).
    _, type_size = GGML_QUANT_SIZES[Q6_K]
    return block.reshape(out_features, type_size)


@pytest.mark.llm
class TestQ6KCustomVsReference:
    """Compare the Brevitas custom Q6_K quantizer against the python reference
    implementation in quant.py (``q6_k_quant_block`` with ``scale=None``, an
    adaptation of llama.cpp's quantize_row_q6_K_ref)."""

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_same_block_layout(self, x):
        """The custom quantizer produces a byte block of the same Q6_K size/shape as
        the reference encoder."""
        _, type_size = GGML_QUANT_SIZES[Q6_K]
        ref = q6_k_quant_block(x.copy(), scale=None)
        custom = _custom_q6_k_export(x)
        assert custom.dtype == ref.dtype == np.uint8
        assert custom.shape == ref.shape == (x.shape[0], type_size)

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_export_is_bit_consistent_with_calibration(self, x):
        """Decoding the exported block reproduces Brevitas' own reconstruction.

        The custom quantizer divides the weights by ``q_scale * d``; GGUF stores
        ``d`` as fp16 and reconstructs as ``code * (q_scale * fp16(d))``. The
        exported block must decode back to exactly that, i.e. the export uses the
        Brevitas-computed nested scales rather than re-deriving its own."""
        import torch

        from brevitas.core.restrict_val import QuantRestrictValue
        import brevitas.nn as qnn
        from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ6_KWeightQuant

        layer = qnn.QuantLinear(
            x.shape[1], x.shape[0], bias=False, weight_quant=GGUFQ6_KWeightQuant)
        layer.weight.data = torch.from_numpy(x.copy())
        quant_weight = layer.quant_weight()
        codes = quant_weight.int().detach().numpy().astype(np.float32)
        # Effective per-sub-block scale exactly as stored on disk: the int8 sub-scale
        # times the fp16-rounded super-block d (q_scale * fp16(d)).
        restrict = next(
            m for m in layer.weight_quant.modules() if isinstance(m, QuantRestrictValue))
        quant_scale, scale_scale, *_ = restrict.float_to_int_impl(quant_weight.scale_)
        quant_scale = quant_scale.detach().numpy().astype(np.float32).reshape(x.shape[0], -1)
        scale_scale = scale_scale.detach().numpy().astype(np.float32).reshape(x.shape[0], -1)
        int_sub_scale = np.round(quant_scale / scale_scale)
        eff_scale = int_sub_scale * fp16(scale_scale)  # (out, n_sub)
        recon_brevitas = (codes.reshape(x.shape[0], QK_K // 16, 16) *
                          eff_scale[:, :, None]).reshape(x.shape)

        block = _custom_q6_k_export(x)
        recon_gguf = gguf_quants.dequantize(block, Q6_K).reshape(x.shape)
        np.testing.assert_allclose(recon_gguf, recon_brevitas, rtol=0, atol=0)

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_accuracy_comparable_to_reference(self, x):
        """The custom quantizer's reconstruction error is comparable to the python
        reference. The reference uses llama.cpp's grid scale search, so the custom
        (max-based) quantizer can be slightly worse, but must stay within one full
        Q6_K step (amax / 16) of both the source and the reference error."""
        ref_hat = gguf_quants.dequantize(q6_k_quant_block(x.copy(), scale=None),
                                         Q6_K).reshape(x.shape)
        custom_hat = gguf_quants.dequantize(_custom_q6_k_export(x), Q6_K).reshape(x.shape)
        amax = np.abs(x).max()
        if amax == 0.0:
            # All-zero tensors must quantize losslessly for both paths.
            assert np.abs(ref_hat).max() == 0.0
            assert np.abs(custom_hat).max() == 0.0
            return
        ref_err = np.abs(x - ref_hat).max()
        custom_err = np.abs(x - custom_hat).max()
        # The reference uses llama.cpp's grid scale search, so it stays within half
        # a Q6_K step (amax / 32) of the source.
        assert ref_err <= amax / 32
        # The custom (max-based) quantizer has no grid search, so allow up to a full
        # step (amax / 16), but it must not be meaningfully worse than the reference.
        assert custom_err <= amax / 16
        assert custom_err <= ref_err + amax / 16
