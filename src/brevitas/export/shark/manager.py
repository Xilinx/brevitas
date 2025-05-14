# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from pathlib import Path

from sharktank.types import Dataset
from sharktank.types import DefaultPrimitiveTensor
from sharktank.types import Theta
import torch
from torch.nn import Module
import torch.nn as nn

from brevitas.export.inference.manager import _override_act_caching_mode
from brevitas.export.inference.manager import _override_bias_caching_mode
from brevitas.export.inference.manager import _override_weight_caching_mode
from brevitas.export.manager import _set_layer_export_handler
from brevitas.export.manager import _set_layer_export_mode
from brevitas.export.manager import _set_proxy_export_handler
from brevitas.export.manager import _set_proxy_export_mode
from brevitas.export.manager import BaseManager
from brevitas.export.shark.handler import SharkActEqualization
from brevitas.export.shark.handler import SharkLinearQuant
from brevitas.export.shark.handler import SharkQuantSDPA
# from brevitas.export.shark.handler import SharkWeightQuant
from brevitas.graph.equalize import EqualizedModule
from brevitas.nn.quant_layer import QuantNonLinearActLayer
from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer


def _quant_wbiol_handler(layer, layer_name, shared_dict):
    if layer.weight_quant.is_quant_enabled:
        _quant_handler(layer, layer_name + '.weight', 'weight_quant', shared_dict)
    else:
        shared_dict[layer_name + '.' + 'weight'] = DefaultPrimitiveTensor(
            name=layer_name + '.' + 'weight', data=layer.weight)
    if layer.input_quant.is_quant_enabled:
        _quant_handler(layer, layer_name + '.q_input', 'input_quant', shared_dict)
    if layer.output_quant.is_quant_enabled:
        _quant_handler(layer, layer_name + '.q_output', 'output_quant', shared_dict)


def _quant_act_handler(layer, layer_name, shared_dict):
    if layer.act_quant.is_quant_enabled:
        _quant_handler(layer, layer_name, 'act_quant', shared_dict)


def _quant_handler(layer, layer_name, quant_name, shared_dict):
    handler = getattr(layer, quant_name).export_handler
    handler.layer_name = layer_name
    handler.shared_dict = shared_dict


# Inheritance from BaseManager is not techincally needed
class SharkManager(BaseManager):
    # handlers = [SharkWeightQuant, SharkActQuant, SharkActFloatQuant, SharkWeightFloatQuant]

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

    def gguf_preprocess(self, model):
        import sys

        import gguf

        from brevitas_examples.llm.gguf_export.convert import ModelBase
        """Export the model to gguf format."""
        output_type = gguf.LlamaFileType.ALL_F32

        config = model.config

        tmp_work_dir = Path('./tmp_dir')
        config.save_pretrained(tmp_work_dir)

        with torch.no_grad():
            hparams = ModelBase.load_hparams(tmp_work_dir)
            model_architecture = hparams["architectures"][0]
            try:
                model_class = ModelBase.from_model_architecture(model_architecture)
            except NotImplementedError:
                sys.exit(1)
            model_class = ModelBase.from_model_architecture(model_architecture)
            model_name = model.name_or_path.split('/')
            if len(model_name[-1]) == 0:
                model_name = model_name[-2]
            else:
                model_name = model_name[-1]

            model_instance = model_class(
                model,
                dir_model=tmp_work_dir,
                ftype=output_type,
                fname_out=tmp_work_dir,
                is_big_endian=False,
                model_name=model_name,
                split_max_tensors=False,
                split_max_size=0,
                dry_run=False,
                small_first_shard=False)
        return model_instance

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
        gguf_model = self.gguf_preprocess(model)
        for k, v in shared_dict.items():
            if k.endswith('.premul_input'):
                prefix = k.removesuffix('.premul_input')
                suffix = '.premul_input'
            elif k.endswith('.q_input'):
                prefix = k.removesuffix('.q_input')
                suffix = '.q_input'
            elif k.endswith('.q_output'):
                prefix = k.removesuffix('.q_output')
                suffix = '.q_output'
            elif k.endswith('.attn_q_output'):
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
            new_k = gguf_model.map_tensor_name(prefix)
            if new_k is None:
                raise

            updated_theta[new_k + suffix] = v
        theta = Theta(updated_theta)
        theta.rename_tensors_to_paths()
        ds = Dataset(self.config, theta)
        ds.save('test_dataset.irpa', io_report_callback=None)
        return ds
