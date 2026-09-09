# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Resolve GGUF converter classes without importing every model module.
"""

from importlib import import_module

from .base import get_model_architecture as get_model_architecture
from .base import ModelBase
from .base import ModelType
from .base import SUPPORTED_OVERRIDE_QTYPES

__all__ = (
    "get_model_architecture",
    "get_model_class",
    "load_all_models",
    "MMPROJ_MODEL_MAP",
    "ModelBase",
    "ModelType",
    "print_registered_models",
    "SUPPORTED_OVERRIDE_QTYPES",
    "TEXT_MODEL_MAP",
)

TEXT_MODEL_MAP: dict[str, str] = {
    'LLaMAForCausalLM': 'llama',
    'LlamaForCausalLM': 'llama',
    'Llama4ForConditionalGeneration': 'llama',
    'LlamaModel': 'llama',
    'LlavaForConditionalGeneration': 'llama',
    'MistralForCausalLM': 'llama',
    'MixtralForCausalLM': 'llama',
    'VLlama3ForCausalLM': 'llama',
    'QWenLMHeadModel': 'qwen',
    'Qwen2AudioForConditionalGeneration': 'qwen',
    'Qwen2ForCausalLM': 'qwen',
    'Qwen2Model': 'qwen',
    'Qwen2MoeForCausalLM': 'qwen',
    'Qwen3ForCausalLM': 'qwen',
    'Qwen3MoeForCausalLM': 'qwen',}

MMPROJ_MODEL_MAP: dict[str, str] = {}


def _load_model_module(name: str, model_type: ModelType) -> None:
    model_map = MMPROJ_MODEL_MAP if model_type == ModelType.MMPROJ else TEXT_MODEL_MAP
    try:
        module_name = model_map[name]
    except KeyError:
        raise NotImplementedError(f"Architecture {name!r} not supported!") from None
    import_module(f"{__name__}.{module_name}")


def get_model_class(name: str, model_type: ModelType = ModelType.TEXT) -> type[ModelBase]:
    _load_model_module(name, model_type)
    return ModelBase.from_model_architecture(name, model_type)


def load_all_models() -> None:
    for module_name in sorted(set(TEXT_MODEL_MAP.values()) | set(MMPROJ_MODEL_MAP.values())):
        import_module(f"{__name__}.{module_name}")


def print_registered_models() -> None:
    load_all_models()
    ModelBase.print_registered_models()
