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
from brevitas_examples.llm.gguf_export.quant import q2_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q3_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_0_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_1_quant_block
from brevitas_examples.llm.gguf_export.quant import q4_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q5_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q6_k_quant_block
from brevitas_examples.llm.gguf_export.quant import q8_0_quant_block

Q4_0 = gguf.GGMLQuantizationType.Q4_0
Q4_1 = gguf.GGMLQuantizationType.Q4_1
Q8_0 = gguf.GGMLQuantizationType.Q8_0
Q2_K = gguf.GGMLQuantizationType.Q2_K
Q3_K = gguf.GGMLQuantizationType.Q3_K
Q4_K = gguf.GGMLQuantizationType.Q4_K
Q5_K = gguf.GGMLQuantizationType.Q5_K
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
    codes = rng.integers(-32, 32, size=(nb, QK_K)).astype(np.float32)  # signed [-32, 31]
    scales = (np.abs(rng.standard_normal((nb, QK_K // 16))) + 0.05).astype(np.float32)
    q = q6_k_quant_block(codes.copy(), scale=scales)
    x_hat = gguf_quants.dequantize(q, Q6_K).reshape(nb, QK_K // 16, 16)
    d, q_scales, _ = _q6_k_quantize_scales(scales)
    eff = fp16(d)[:, None] * q_scales.astype(np.float32)  # effective per-sub-block scale
    expected = eff[:, :, None] * codes.reshape(nb, QK_K // 16, 16)
    np.testing.assert_allclose(x_hat, expected, rtol=0, atol=0)


def _custom_q6_k_export(weight: np.ndarray, weight_quant=None) -> np.ndarray:
    """Quantize ``weight`` with a Brevitas custom Q6_K quantizer and pack it to a
    GGUF Q6_K block exactly as brevitas_examples...convert.ModelBase.quantize does.

    ``weight_quant`` defaults to the absmax-based ``GGUFQ6_KWeightQuant``; pass
    ``GGUFQ6_KWeightQuantMSE`` to exercise the MSE scale search variant.

    Returns the packed uint8 block (one row of blocks per weight row).
    """
    import torch

    from brevitas.core.restrict_val import QuantRestrictValue
    import brevitas.nn as qnn
    from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ6_KWeightQuant
    from brevitas_examples.llm.gguf_export.quant import ggml_quant

    if weight_quant is None:
        weight_quant = GGUFQ6_KWeightQuant
    out_features, in_features = weight.shape
    layer = qnn.QuantLinear(in_features, out_features, bias=False, weight_quant=weight_quant)
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


@pytest.mark.llm
class TestQ6KMSEScaleSearch:
    """The MSE Q6_K quantizer (``GGUFQ6_KWeightQuantMSE``) uses Brevitas' MSE grid
    search to pick the per-sub-block scale instead of plain absmax."""

    def _mse_quant(self):
        from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ6_KWeightQuantMSE
        return GGUFQ6_KWeightQuantMSE

    def test_mse_iters_is_configured(self):
        """The MSE variant sets a non-default number of grid-search iterations and
        actually forwards it to the MSE module (Brevitas would otherwise silently
        use its default of 20)."""
        import brevitas.core.stats.stats_op as stats_op
        import brevitas.nn as qnn
        from brevitas_examples.llm.gguf_export.base_quantizers import Q6_K_MSE_ITERS

        seen = {}
        original_init = stats_op.MSE.__init__

        def spy_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            seen['num'] = self.num

        stats_op.MSE.__init__ = spy_init
        try:
            layer = qnn.QuantLinear(256, 8, bias=False, weight_quant=self._mse_quant())
            layer.weight.data = torch.from_numpy(_normal(0, 8).astype(np.float32))
            layer.quant_weight()
        finally:
            stats_op.MSE.__init__ = original_init
        assert seen.get('num') == Q6_K_MSE_ITERS
        assert Q6_K_MSE_ITERS != 20  # must differ from Brevitas' default to be meaningful

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_valid_q6_k_block(self, x):
        """The MSE variant produces a valid Q6_K block: right layout and every element
        within one full Q6_K step (amax / 16) of the source."""
        _, type_size = GGML_QUANT_SIZES[Q6_K]
        block = _custom_q6_k_export(x, weight_quant=self._mse_quant())
        assert block.dtype == np.uint8
        assert block.shape == (x.shape[0], type_size)
        x_hat = gguf_quants.dequantize(block, Q6_K).reshape(x.shape)
        amax = np.abs(x).max()
        if amax == 0.0:
            assert np.abs(x_hat).max() == 0.0
        else:
            assert np.abs(x - x_hat).max() <= amax / 16

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_not_worse_than_absmax(self, x):
        """MSE searches candidate scales evaluated through the full nested Q6_K quant
        and keeps the lowest-L2 one, so its MSE reconstruction error is never worse
        (up to fp16 super-block rounding) than the plain absmax quantizer."""
        mse_hat = gguf_quants.dequantize(
            _custom_q6_k_export(x, weight_quant=self._mse_quant()), Q6_K).reshape(x.shape)
        max_hat = gguf_quants.dequantize(_custom_q6_k_export(x), Q6_K).reshape(x.shape)
        amax = np.abs(x).max()
        if amax == 0.0:
            return
        mse_l2 = np.mean((x - mse_hat) ** 2)
        max_l2 = np.mean((x - max_hat) ** 2)
        # Allow a small slack for fp16 storage of the super-block scale.
        assert mse_l2 <= max_l2 + (amax / 32) ** 2


def _custom_q5_k_export(weight: np.ndarray) -> np.ndarray:
    """Quantize ``weight`` with the Brevitas custom Q5_K quantizer and pack it to a
    GGUF Q5_K block exactly as brevitas_examples...convert.ModelBase.quantize does.

    Returns the packed uint8 block (one row of blocks per weight row).
    """
    import torch

    from brevitas.core.restrict_val import QuantRestrictValue
    from brevitas.core.zero_point import _ScaleShiftQuantZeroPoint
    import brevitas.nn as qnn
    from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ5_KWeightQuant
    from brevitas_examples.llm.gguf_export.quant import ggml_quant

    out_features, in_features = weight.shape
    layer = qnn.QuantLinear(in_features, out_features, bias=False, weight_quant=GGUFQ5_KWeightQuant)
    layer.weight.data = torch.from_numpy(weight.copy())
    quant_weight = layer.quant_weight()
    quant_data = quant_weight.int()
    scale = quant_weight.scale_ if hasattr(quant_weight, 'scale_') else quant_weight.scale
    zp = quant_weight.zero_point_ if hasattr(
        quant_weight, 'zero_point_') else quant_weight.zero_point
    weight_quant = layer.weight_quant
    # Read the calibrated nested scale/zero-point, mirroring convert.py's Q4_K/Q5_K branch.
    restrict = next(m for m in weight_quant.modules() if isinstance(m, QuantRestrictValue))
    quant_scale, scale_scale, *_ = restrict.float_to_int_impl(scale)
    scale_shift = next(
        m for m in weight_quant.modules() if isinstance(m, _ScaleShiftQuantZeroPoint))
    quant_zp, scale_zp, *_ = scale_shift.zp_int_quant(zp * scale)
    block = ggml_quant(
        quant_data,
        Q5_K,
        quant_scale,
        quant_zp,
        wmin_m=quant_zp,
        d_scale=scale_scale,
        d_wmin_m=scale_zp)
    _, type_size = GGML_QUANT_SIZES[Q5_K]
    return block.reshape(out_features, type_size)


@pytest.mark.llm
class TestQ5KCustom:
    """The Brevitas custom Q5_K quantizer packs to a valid GGUF Q5_K block whose
    decoded values match Brevitas' own reconstruction (bit-consistent export)."""

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_block_layout(self, x):
        _, type_size = GGML_QUANT_SIZES[Q5_K]
        block = _custom_q5_k_export(x)
        assert block.dtype == np.uint8
        assert block.shape == (x.shape[0], type_size)

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_export_is_bit_consistent_with_calibration(self, x):
        """Decoding the exported block reproduces Brevitas' own reconstruction, up to
        the fp16 rounding of the super-block d / dmin factors stored on disk."""
        import torch

        import brevitas.nn as qnn
        from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ5_KWeightQuant

        layer = qnn.QuantLinear(
            x.shape[1], x.shape[0], bias=False, weight_quant=GGUFQ5_KWeightQuant)
        layer.weight.data = torch.from_numpy(x.copy())
        recon_brevitas = layer.quant_weight().value.detach().numpy()
        recon_gguf = gguf_quants.dequantize(_custom_q5_k_export(x), Q5_K).reshape(x.shape)
        amax = np.abs(x).max()
        # fp16 super-scale storage introduces at most a small relative error.
        atol = 1e-2 * amax if amax > 0 else 1e-6
        np.testing.assert_allclose(recon_gguf, recon_brevitas, rtol=0, atol=atol)

    @pytest_cases.parametrize("x", list(MODEL_TENSORS.values()), ids=list(MODEL_TENSORS))
    def test_valid_q5_k_block(self, x):
        """Every element lands within one full Q5_K step of the source."""
        x_hat = gguf_quants.dequantize(_custom_q5_k_export(x), Q5_K).reshape(x.shape)
        amax = np.abs(x).max()
        if amax == 0.0:
            assert np.abs(x_hat).max() == 0.0
        else:
            # 5-bit asymmetric grid spans the block range in 31 steps.
            assert np.abs(x - x_hat).max() <= amax / 15


@pytest.mark.llm
def test_gguf_q5_k_registered():
    """The gguf_q5_k custom quantizer is registered and keeps the first/last layer at
    Q6_K (per llama.cpp), Q5_K elsewhere."""
    import torch

    from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
    from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ5_KWeightQuant
    from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ6_KWeightQuant
    import brevitas_examples.llm.gguf_export.custom_quantizers  # noqa: F401  (registers)

    assert "gguf_q5_k" in QUANTIZERS_REGISTRY.get_registered_keys()
    quantizer = QUANTIZERS_REGISTRY.get("gguf_q5_k")
    weight_quant = quantizer.weight_quant
    embedding = torch.nn.Embedding(8, 8)
    linear = torch.nn.Linear(8, 8)
    assert weight_quant(embedding, "model.embed_tokens") is GGUFQ6_KWeightQuant
    assert weight_quant(linear, "model.lm_head") is GGUFQ6_KWeightQuant
    assert weight_quant(linear, "model.layers.0.mlp.gate_proj") is GGUFQ5_KWeightQuant
