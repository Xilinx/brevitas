# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn.functional as F
from transformers.integrations.executorch import TorchExportableModuleForDecoderOnlyLM
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


class make_dynamo_compatible:

    def __init__(self, model):
        self.model = model
        self.model_config = model.config
        if hasattr(self.model.generation_config, 'cache_implementation'):
            self.model_cache_implementation = self.model.generation_config.cache_implementation
        else:
            self.model_cache_implementation = None

    def __enter__(self):
        # We set cache_implementation to `static` for compatibility with dynamo
        self.model.generation_config.cache_implementation = "static"
        # Because getattr does not fall back to default with `config` class, we need to manually fill
        # `head_dim` if it is None
        # https://github.com/huggingface/transformers/blob/47b0e478f324b54f177ea7998a0791870fdd0324/src/transformers/integrations/executorch.py#L538
        if not hasattr(self.model.config, 'head_dim') or self.model.config.head_dim is None:
            self.model.config.head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        # Wrapping the model applies certain patches to make it work with dynamo,
        # but then we can unwrap it immediately.
        # We need to specify batch_size and max_cache_len. The latter is not important since we disable
        # cache anyway while we trace the model.
        self.model = TorchExportableModuleForDecoderOnlyLM(
            self.model, batch_size=1, max_cache_len=1).model.model
        # Caching should be disabled to make it work with dynamo
        # The other alternative is to use static_cache
        self.model.config.use_cache = False
        return self

    def __exit__(self, *args, **kwargs):
        # Restore configuration
        self.model.config = self.model_config
        if self.model_cache_implementation is not None:
            self.model.generation_config.cache_implementation = self.model_cache_implementation
