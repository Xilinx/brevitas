# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from brevitas.graph import TorchFunctionalToModule
from brevitas.nn import ScaledDotProductAttention
from brevitas.utils.logging import setup_logger

logging = setup_logger(__name__)


def replace_sdpa_with_quantizable_layers(model, is_fx=True, eager_quant_sdpa_class=None):
    if is_fx:
        fn_to_module_map = ((F.scaled_dot_product_attention, ScaledDotProductAttention),)
        model = TorchFunctionalToModule(fn_to_module_map=fn_to_module_map).apply(model)
    else:
        # We rely on the following:
        # - Attention functions accepts the current module as input
        # - We can add a new entry in the dict of supported attention functions
        # - Attention Modules' name end with `Attention`. The user can also override this

        from brevitas_examples.llm.llm_quant.mha_layers import quant_sdpa_attention_forward
        ALL_ATTENTION_FUNCTIONS['quant_sdpa'] = quant_sdpa_attention_forward
        model.config._attn_implementation = 'quant_sdpa'
        for n, m in model.named_modules():
            if eager_quant_sdpa_class == 'auto':
                if type(m).__name__.lower().endswith('attention'):
                    quant_block_type = type(m)
                    break
            else:
                if type(m).__name__.lower() == eager_quant_sdpa_class.lower():
                    quant_block_type = type(m)
                    break
        logging.info(f"Attention module is {quant_block_type}")
        for m in model.modules():
            if isinstance(m, quant_block_type):
                m.attn = ScaledDotProductAttention()

    return model


@torch.no_grad()
def add_zero_bias_to_linear(model: torch.nn.Module) -> torch.nn.Module:
    for name, module in model.named_modules():
        if type(module) == torch.nn.Linear:
            if module.bias is None:
                module.register_parameter(
                    "bias",
                    torch.nn.Parameter(
                        torch.zeros((module.weight.shape[0],),
                                    device=module.weight.device,
                                    dtype=module.weight.dtype)),
                )
    return model
