# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import torch
import torch.nn.functional as F

from brevitas import torch_version
from brevitas.nn import QuantScaledDotProductAttention
from brevitas.nn import ScaledDotProductAttention
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat
from tests.marker import requires_pt_ge

ATOL = 1e-6
EMBED_DIM = 9
HEAD_DIM = 3
BATCH_SIZE = 2
SEQUENCE_LENGTH = 4
PAST_SEQUENCE_LENGTH = 5
DROPOUT_SEED = 42


class TestScaledDotProductAttention:

    @requires_pt_ge('2.0')
    # Check what kwargs are properly filtered and override defaults
    def test_sdpa_init(self):
        extra_kwargs = {
            'softmax_input_bit_width': 2,
            'attn_output_weights_bit_width': 3,
            'q_scaled_bit_width': 4,
            'k_transposed_bit_width': 5,
            'v_bit_width': 6,
            'sdpa_output_bit_width': 7,}
        qm = QuantScaledDotProductAttention(
            softmax_input_quant=Int8ActPerTensorFloat,
            attn_output_weights_quant=Uint8ActPerTensorFloat,
            q_scaled_quant=Int8ActPerTensorFloat,
            k_transposed_quant=Int8ActPerTensorFloat,
            v_quant=Int8ActPerTensorFloat,
            sdpa_output_quant=Int8ActPerTensorFloat,
            **extra_kwargs,
        )

        # Check that the `kwargs` have been applied correctly
        prefixes = ["softmax_input", "attn_output_weights", "q_scaled", "v", "sdpa_output"]
        for k in extra_kwargs.keys():
            checked = False
            if "softmax_input_" in k:
                assert int(qm.softmax_input_quant.act_quant.bit_width().item()) == extra_kwargs[k]
                checked = True
            elif "attn_output_weights_" in k:
                assert int(
                    qm.attn_output_weights_quant.act_quant.bit_width().item()) == extra_kwargs[k]
                checked = True
            elif "q_scaled_" in k:
                assert int(qm.q_scaled_quant.act_quant.bit_width().item()) == extra_kwargs[k]
                checked = True
            elif "k_transposed_" in k:
                assert int(qm.k_transposed_quant.act_quant.bit_width().item()) == extra_kwargs[k]
                checked = True
            elif "v_" in k:
                assert int(qm.v_quant.act_quant.bit_width().item()) == extra_kwargs[k]
                checked = True
            elif "sdpa_output_" in k:
                assert int(qm.sdpa_output_quant.act_quant.bit_width().item()) == extra_kwargs[k]
                checked = True
            assert checked, f"Unmatched kwarg: {k}"

    @requires_pt_ge('2.0')
    @pytest.mark.parametrize("dropout_p", [0.0, 0.5])
    @pytest.mark.parametrize("is_causal", [True, False])
    @pytest.mark.parametrize("scale", [None, 0.3])
    @pytest.mark.parametrize("enable_gqa", [False, True])
    @pytest.mark.parametrize("rand_attn_mask", [False, True])
    # Sanity check, since `ScaledDotProductAttention` just calls `F.scaled_dot_product_attention` in its forward function
    def test_sdpa_fwd(self, dropout_p, is_causal, scale, enable_gqa, rand_attn_mask):
        extra_kwargs = {
            "dropout_p": dropout_p,
            "is_causal": is_causal,
            "scale": scale,
            "enable_gqa": enable_gqa,}
        if torch_version < version.parse('2.5.0'):
            del extra_kwargs["enable_gqa"]
        if torch_version < version.parse('2.1.0'):
            del extra_kwargs["scale"]

        kv_length = PAST_SEQUENCE_LENGTH + SEQUENCE_LENGTH
        m = ScaledDotProductAttention()
        q = torch.randn(BATCH_SIZE, HEAD_DIM, SEQUENCE_LENGTH, EMBED_DIM)
        k = torch.randn(BATCH_SIZE, HEAD_DIM, kv_length, EMBED_DIM)
        v = torch.randn(BATCH_SIZE, HEAD_DIM, kv_length, EMBED_DIM)
        if rand_attn_mask and not is_causal:
            attn_mask = torch.randint(
                low=0, high=2, size=(BATCH_SIZE, 1, SEQUENCE_LENGTH, kv_length), dtype=torch.bool)
        else:
            attn_mask = None
        if dropout_p > 0.0:
            torch.manual_seed(DROPOUT_SEED)
        ref_out = F.scaled_dot_product_attention(q, k, v, attn_mask, **extra_kwargs)
        if dropout_p > 0.0:
            torch.manual_seed(DROPOUT_SEED)
        out = m(q, k, v, attn_mask, **extra_kwargs)
        assert torch.isclose(out, ref_out, atol=ATOL).all()
        assert torch.isclose(out, ref_out, atol=ATOL).all()

    @requires_pt_ge('2.0')
    @pytest.mark.parametrize("dropout_p", [0.0, 0.5])
    @pytest.mark.parametrize("is_causal", [True, False])
    @pytest.mark.parametrize("scale", [None, 0.3])
    @pytest.mark.parametrize("enable_gqa", [False, True])
    @pytest.mark.parametrize("rand_attn_mask", [False, True])
    def test_sdpa_quant_disabled_fwd(self, dropout_p, is_causal, scale, enable_gqa, rand_attn_mask):
        extra_kwargs = {
            "dropout_p": dropout_p,
            "is_causal": is_causal,
            "scale": scale,
            "enable_gqa": enable_gqa,}
        if torch_version < version.parse('2.5.0'):
            del extra_kwargs["enable_gqa"]
        if torch_version < version.parse('2.1.0'):
            del extra_kwargs["scale"]

        kv_length = PAST_SEQUENCE_LENGTH + SEQUENCE_LENGTH
        m = ScaledDotProductAttention()
        qm = QuantScaledDotProductAttention(
            softmax_input_quant=None,
            attn_output_weights_quant=None,
            q_scaled_quant=None,
            k_transposed_quant=None,
            v_quant=None,
            sdpa_output_quant=None,
        )
        q = torch.randn(BATCH_SIZE, HEAD_DIM, SEQUENCE_LENGTH, EMBED_DIM)
        k = torch.randn(BATCH_SIZE, HEAD_DIM, kv_length, EMBED_DIM)
        v = torch.randn(BATCH_SIZE, HEAD_DIM, kv_length, EMBED_DIM)
        if rand_attn_mask and not is_causal:
            attn_mask = torch.randint(
                low=0, high=2, size=(BATCH_SIZE, 1, SEQUENCE_LENGTH, kv_length), dtype=torch.bool)
        else:
            attn_mask = None
        if dropout_p > 0.0:
            torch.manual_seed(DROPOUT_SEED)
        ref_out = m(q, k, v, attn_mask, **extra_kwargs)
        if dropout_p > 0.0:
            torch.manual_seed(DROPOUT_SEED)
        out = qm(q, k, v, attn_mask, **extra_kwargs)
        assert torch.isclose(out, ref_out, atol=ATOL).all()
        assert torch.isclose(out, ref_out, atol=ATOL).all()
