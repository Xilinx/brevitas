# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
GGUF-standard custom quantizers.

These plug into the LLM entry point via ``--custom-quantizer <name>`` and the
:data:`brevitas_examples.common.generative.quantizers.QUANTIZERS_REGISTRY`. Each
class selects a GGUF base weight quantizer for the linear layers.

First/last-layer handling follows llama.cpp's ``llama_tensor_get_type_impl``
(see src/llama-quant.cpp): the high-impact ``token_embd`` (embedding) and
``output`` (lm_head) tensors are kept at a higher precision than the rest of the
model. For the 4-bit recipes (Q4_0/Q4_1/Q4_K) llama.cpp bumps the output tensor
(and tied token embeddings) to Q6_K, so we do the same here. For Q6_K and Q8_0
the whole model is already at that precision, so no separate bump is applied.

The embedding/last-layer quantizers are only used when the entry point is run
with ``--quantize-first-last-layer``; otherwise those layers are left
unquantized (the lambda result is ignored for them).
"""

import torch

from brevitas.utils.python_utils import Registry
from brevitas_examples.common.generative.quantizers import BaseQuantizer
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY

from .base_quantizers import GGUFQ4_0WeightQuant
from .base_quantizers import GGUFQ4_1WeightQuant
from .base_quantizers import GGUFQ4_KWeightQuant
from .base_quantizers import GGUFQ6_KWeightQuant
from .base_quantizers import GGUFQ8_0WeightQuant

# Names of the high-impact first/last tensors that llama.cpp keeps at higher
# precision (token_embd / output, a.k.a. lm_head / embed_out).
_LAST_LAYER_NAMES = ("lm_head", "embed_out", "output")


def is_first_or_last_layer(module, name):
    # ``name`` is the fully-qualified module path (e.g. ``model.lm_head``), so we
    # compare the last component.
    short_name = name.split(".")[-1] if name is not None else ""
    # First layer: the token embedding.
    if isinstance(module, torch.nn.Embedding):
        return True
    # Last layer: the output projection / lm_head.
    if short_name in _LAST_LAYER_NAMES:
        return True
    # Optionally we can check whether input/output dim == vocab_size.
    return False


def _high_precision_for(base_weight_quant, high_precision_weight_quant):
    # Returns a per-module weight_quant lambda that bumps the first/last layer to
    # ``high_precision_weight_quant`` and uses ``base_weight_quant`` elsewhere.
    return lambda module, name: (
        high_precision_weight_quant if is_first_or_last_layer(module, name) else base_weight_quant)


@Registry.register(QUANTIZERS_REGISTRY, "gguf_q4_0")
class GGUFQ4_0(BaseQuantizer):
    """GGUF Q4_0: signed 4-bit groups of 32; first/last layer kept at Q6_K."""
    weight_quant = _high_precision_for(GGUFQ4_0WeightQuant, GGUFQ6_KWeightQuant)


@Registry.register(QUANTIZERS_REGISTRY, "gguf_q4_1")
class GGUFQ4_1(BaseQuantizer):
    """GGUF Q4_1: asymmetric 4-bit groups of 32; first/last layer kept at Q6_K."""
    weight_quant = _high_precision_for(GGUFQ4_1WeightQuant, GGUFQ6_KWeightQuant)


@Registry.register(QUANTIZERS_REGISTRY, "gguf_q4_k")
class GGUFQ4_K(BaseQuantizer):
    """GGUF Q4_K: 4-bit super-blocks with nested scales/mins; first/last at Q6_K."""
    weight_quant = _high_precision_for(GGUFQ4_KWeightQuant, GGUFQ6_KWeightQuant)


@Registry.register(QUANTIZERS_REGISTRY, "gguf_q6_k")
class GGUFQ6_K(BaseQuantizer):
    """GGUF Q6_K: signed 6-bit super-blocks with nested scales (uniform)."""
    weight_quant = GGUFQ6_KWeightQuant


@Registry.register(QUANTIZERS_REGISTRY, "gguf_q8_0")
class GGUFQ8_0(BaseQuantizer):
    """GGUF Q8_0: signed 8-bit groups of 32 (uniform)."""
    weight_quant = GGUFQ8_0WeightQuant
