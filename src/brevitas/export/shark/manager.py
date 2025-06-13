# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable

import gguf
from sharktank.types import Dataset
from sharktank.types import DefaultPrimitiveTensor
from sharktank.types import Theta
import torch
from torch.nn import Module
import torch.nn as nn

from brevitas.export.manager import _set_layer_export_handler
from brevitas.export.manager import _set_layer_export_mode
from brevitas.export.manager import BaseManager
from brevitas.export.shark.handler import SharkActEqualization
from brevitas.export.shark.handler import SharkLinearQuant
from brevitas.export.shark.handler import SharkQuantSDPA
from brevitas_examples.llm.gguf_export.convert import ModelBase


def _optional_int_prop(p: dict[str, Any], name: str, default_value: int) -> int:
    value = p.get(name, default_value)
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e


def _float_prop(p: dict[str, Any], name: str) -> float:
    try:
        return float(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a float and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _get_dataset_props(config_json_struct) -> dict:
    # Separate meta parameters (prefixed with _) from hparams.
    meta_params = {k: v for k, v in config_json_struct.__dict__.items() if k.startswith("_")}
    hparams = {k: v for k, v in config_json_struct.__dict__.items() if not k.startswith("_")}
    return {
        "meta": meta_params,
        "hparams": hparams,}


def _int_prop(p: dict[str, Any], name: str) -> int:
    try:
        return int(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def convert_hf_hparams_to_gguf(hf_hparams: dict[str, any]) -> dict[str, any]:
    hp = hf_hparams["hparams"]
    attention_head_count = _int_prop(hp, "num_attention_heads")
    attn_head_dim = int(_int_prop(hp, "hidden_size") // _int_prop(hp, "num_attention_heads"))
    attn_head_dim = int(_optional_int_prop(hp, "head_dim", attn_head_dim))
    return {
        "llama.context_length":
            _int_prop(hp, "max_position_embeddings"),
        "llama.embedding_length":
            _int_prop(hp, "hidden_size"),
        "llama.block_count":
            _int_prop(hp, "num_hidden_layers"),
        "llama.feed_forward_length":
            _int_prop(hp, "intermediate_size"),
        "llama.rope.dimension_count":
            attn_head_dim,
        "llama.attention.head_count":
            attention_head_count,
        "llama.attention.layer_norm_rms_epsilon":
            _float_prop(hp, "rms_norm_eps"),
        "llama.attention.head_count_kv":
            _optional_int_prop(hp, "num_key_value_heads", attention_head_count),}


def _get_dataset_props(config_json_struct) -> dict:
    # Separate meta parameters (prefixed with _) from hparams.
    meta_params = {k: v for k, v in config_json_struct.items() if k.startswith("_")}
    hparams = {k: v for k, v in config_json_struct.items() if not k.startswith("_")}
    return {
        "meta": meta_params,
        "hparams": hparams,}


def find_hparam(keys: Iterable[str], hparams: Dict[str, int], optional: bool = False) -> Any:
    key = next((k for k in keys if k in hparams), None)
    if key is not None:
        return hparams[key]
    if optional:
        return None
    raise KeyError(f"could not find any of: {keys}")


# Inheritance from BaseManager is not techincally needed
class SharkManager(BaseManager):
    handlers = [SharkActEqualization, SharkLinearQuant, SharkQuantSDPA]

    def __init__(self, config=None):
        super().__init__()
        if config == None:
            config = dict()
        self.config = config

    @classmethod
    def set_export_mode(cls, model: Module, enabled: bool):
        _set_layer_export_mode(model, enabled)

    @classmethod
    def set_export_handler(cls, module: Module, name: str, shared_dict: dict):
        _set_layer_export_handler(cls, module)
        if hasattr(module, 'export_handler') and module.export_handler is not None:
            module.export_handler.layer_name = name
            module.export_handler.shared_dict = shared_dict

    def gguf_tensor_map(self):

        hf_arch = self.config.to_dict()["architectures"][0]
        gguf_arch = ModelBase.from_model_architecture(hf_arch)
        block_count = find_hparam(["n_layers", "num_hidden_layers", "n_layer", "num_layers"],
                                  self.config.to_dict())
        tensor_map = gguf.get_tensor_name_map(gguf_arch.model_arch, block_count)
        return tensor_map

    def export(self, model, *model_args, **model_kwargs):
        from brevitas_examples.llm.main import offload_model
        from brevitas_examples.llm.main import remove_hooks

        shared_dict = {}

        for name, module in model.named_modules():
            self.set_export_handler(module, name, shared_dict)
        self.set_export_mode(model, enabled=True)

        print("Forward starts")
        offload_model(model)
        with torch.no_grad():
            model(*model_args, **model_kwargs)
        remove_hooks(model)
        print("Forward ends")

        self.set_export_mode(model, enabled=False)

        updated_theta = dict()
        tensor_map = self.gguf_tensor_map()
        for k, v in shared_dict.items():
            # if k.endswith('.premul_input'):
            #     prefix = k.removesuffix('.premul_input')
            #     suffix = '.premul_input'
            # elif k.endswith('.q_input'):
            #     prefix = k.removesuffix('.q_input')
            #     suffix = '.q_input'
            # elif k.endswith('.q_output'):
            #     prefix = k.removesuffix('.q_output')
            #     suffix = '.q_output'
            if k.endswith('.attn_q_output'):
                prefix = k.removesuffix('.attn_q_output') + '.q_proj'
                suffix = '.q_output'
            elif k.endswith('.attn_k_output'):
                prefix = k.removesuffix('.attn_k_output') + '.k_proj'
                suffix = '.q_output'
            elif k.endswith('.attn_v_output'):
                prefix = k.removesuffix('.attn_v_output') + '.v_proj'
                suffix = '.q_output'
            else:
                prefix = k
                suffix = ''
            new_k = tensor_map.get_name(
                prefix, try_suffixes=(".weight", ".bias", ".premul_input", ".q_input", ".q_output"))
            if new_k is None:
                raise

            updated_theta[new_k + suffix] = v
        theta = Theta(updated_theta)
        theta.rename_tensors_to_paths()
        ds = Dataset(convert_hf_hparams_to_gguf(_get_dataset_props(self.config.to_dict())), theta)
        ds.save('test_dataset.irpa', io_report_callback=None)
        return ds
