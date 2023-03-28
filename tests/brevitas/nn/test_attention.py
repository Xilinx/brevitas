# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from packaging import version
import pytest
import torch
from torch.nn import MultiheadAttention

from brevitas import torch_version
from brevitas.nn import QuantMultiheadAttention

ATOL = 1e-6
EMBED_DIM = 9
NUM_HEADS = 3


class TestQuantMultiheadAttention:

    @pytest.mark.parametrize("batch_first", [True, False])
    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("packed_in_proj", [True, False])
    def test_mha_quant_disabled_fwd(self, batch_first, bias, packed_in_proj):
        extra_kwargs = {}
        if torch_version >= version.parse('1.9.1'):
            extra_kwargs['batch_first'] = batch_first

        m = MultiheadAttention(EMBED_DIM, NUM_HEADS, bias=bias, **extra_kwargs)
        qm = QuantMultiheadAttention(
            EMBED_DIM,
            NUM_HEADS,
            packed_in_proj=packed_in_proj,
            in_proj_input_quant=None,
            in_proj_weight_quant=None,
            in_proj_bias_quant=None,
            softmax_input_quant=None,
            attn_output_weights_quant=None,
            q_scaled_quant=None,
            k_transposed_quant=None,
            v_quant=None,
            out_proj_input_quant=None,
            out_proj_weight_quant=None,
            out_proj_bias_quant=None,
            out_proj_output_quant=None,
            bias=bias,
            **extra_kwargs)
        qm.load_state_dict(m.state_dict())
        inp = torch.randn(2, 5, EMBED_DIM)
        ref_out = m(inp, inp, inp)
        out = qm(inp, inp, inp)
        assert torch.isclose(out[0], ref_out[0], atol=ATOL).all()
        assert torch.isclose(out[1], ref_out[1], atol=ATOL).all()
