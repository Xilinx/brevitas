# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

from brevitas import torch_version
from brevitas.nn import QuantScaledDotProductAttention
from brevitas.nn import ScaledDotProductAttention
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
    @pytest.mark.parametrize("dropout_p", [0.0, 0.5])
    @pytest.mark.parametrize("is_causal", [True, False])
    @pytest.mark.parametrize("scale", [None, 0.3])
    @pytest.mark.parametrize("enable_gqa", [False, True])
    @pytest.mark.parametrize("rand_attn_mask", [False, True])
    # Sanity check, since `ScaledDotProductAttention` just called `F.scaled_dot_product_attention` in its forward function
    def test_sdpa_fwd(self, dropout_p, is_causal, scale, enable_gqa, rand_attn_mask):
        extra_kwargs = {
            "dropout_p": dropout_p,
            "is_causal": is_causal,
            "scale": scale,
            "enable_gqa": enable_gqa,}
        if torch_version < version.parse('2.5.0'):
            del extra_kwargs["enable_gqa"]

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
        ref_out = scaled_dot_product_attention(q, k, v, attn_mask, **extra_kwargs)
        if dropout_p > 0.0:
            torch.manual_seed(DROPOUT_SEED)
        out = m(q, k, v, attn_mask, **extra_kwargs)
        assert torch.isclose(out, ref_out, atol=ATOL).all()
        assert torch.isclose(out, ref_out, atol=ATOL).all()
