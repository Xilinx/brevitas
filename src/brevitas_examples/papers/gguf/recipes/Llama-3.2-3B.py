# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Per-model GGUF mixed-precision recipes for Llama 3.2 3B (base or Instruct).

The per-layer bump rules in ``_Q4_0_RECIPE``, ``_Q4_K_S_RECIPE``, and
``_Q4_K_M_RECIPE`` (used by ``gguf_q4_0``, ``gguf_q4_k_s``, ``gguf_q4_k_m``,
and ``gguf_q5_k_m``) are adapted from
llama.cpp's ``llama_tensor_get_type_impl`` in
https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp,
released under the following LICENSE:

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Load via::

    --custom-quantizer=/path/to/Llama-3.2-3B.py:gguf_q4_k_m
"""

from brevitas.utils.python_utils import Registry
from brevitas_examples.common.generative.quantizers import BaseQuantizer
from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ4_0WeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ4_1WeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ4_KWeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ5_KWeightQuant
from brevitas_examples.llm.gguf_export.base_quantizers import GGUFQ6_KWeightQuant
from brevitas_examples.llm.gguf_export.custom_quantizers import is_first_or_last_layer
from brevitas_examples.papers.gguf.recipes.common import RecipeMixin

_MODEL_NAME = "Llama-3.2-3B"

# Q4_0: ffn_down on layers 0..(n_layer/8 - 1) bumped to Q4_1 (n_layer=28 -> 0-2).
_Q4_0_RECIPE = {
    "model.layers.0.mlp.down_proj": GGUFQ4_1WeightQuant,
    "model.layers.1.mlp.down_proj": GGUFQ4_1WeightQuant,
    "model.layers.2.mlp.down_proj": GGUFQ4_1WeightQuant,}

# Q4_K_S: attn_v layers 0-3 and ffn_down layers 0-2 bumped to Q5_K.
_Q4_K_S_RECIPE = {
    "model.layers.0.self_attn.v_proj": GGUFQ5_KWeightQuant,
    "model.layers.1.self_attn.v_proj": GGUFQ5_KWeightQuant,
    "model.layers.2.self_attn.v_proj": GGUFQ5_KWeightQuant,
    "model.layers.3.self_attn.v_proj": GGUFQ5_KWeightQuant,
    "model.layers.0.mlp.down_proj": GGUFQ5_KWeightQuant,
    "model.layers.1.mlp.down_proj": GGUFQ5_KWeightQuant,
    "model.layers.2.mlp.down_proj": GGUFQ5_KWeightQuant,}

# Q4_K_M / Q5_K_M: attn_v and ffn_down on these 14 layers bumped to Q6_K.
_Q4_K_M_RECIPE = {
    "model.layers.0.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.1.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.2.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.5.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.8.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.11.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.14.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.17.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.20.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.23.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.24.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.25.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.26.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.27.self_attn.v_proj": GGUFQ6_KWeightQuant,
    "model.layers.0.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.1.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.2.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.5.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.8.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.11.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.14.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.17.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.20.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.23.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.24.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.25.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.26.mlp.down_proj": GGUFQ6_KWeightQuant,
    "model.layers.27.mlp.down_proj": GGUFQ6_KWeightQuant,}


@Registry.register(QUANTIZERS_REGISTRY, "gguf_q4_0")
class GGUFQ4_0(RecipeMixin, BaseQuantizer):
    expected_model_name = _MODEL_NAME
    weight_quant = lambda module, name: (
        GGUFQ6_KWeightQuant if is_first_or_last_layer(module, name) else
        _Q4_0_RECIPE.get(name, GGUFQ4_0WeightQuant))


@Registry.register(QUANTIZERS_REGISTRY, "gguf_q4_k_s")
class GGUFQ4_K_S(RecipeMixin, BaseQuantizer):
    expected_model_name = _MODEL_NAME
    weight_quant = lambda module, name: (
        GGUFQ6_KWeightQuant if is_first_or_last_layer(module, name) else
        _Q4_K_S_RECIPE.get(name, GGUFQ4_KWeightQuant))


@Registry.register(QUANTIZERS_REGISTRY, "gguf_q4_k_m")
class GGUFQ4_K_M(RecipeMixin, BaseQuantizer):
    expected_model_name = _MODEL_NAME
    weight_quant = lambda module, name: (
        GGUFQ6_KWeightQuant if is_first_or_last_layer(module, name) else
        _Q4_K_M_RECIPE.get(name, GGUFQ4_KWeightQuant))


@Registry.register(QUANTIZERS_REGISTRY, "gguf_q5_k_m")
class GGUFQ5_K_M(RecipeMixin, BaseQuantizer):
    expected_model_name = _MODEL_NAME
    weight_quant = lambda module, name: (
        GGUFQ6_KWeightQuant if is_first_or_last_layer(module, name) else
        _Q4_K_M_RECIPE.get(name, GGUFQ5_KWeightQuant))
