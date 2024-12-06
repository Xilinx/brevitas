# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import warnings

from packaging import version
import torch
import torch.nn.functional as F
import transformers
from transformers.models.opt.modeling_opt import OPTAttention

from brevitas.graph import ModuleToModuleByClass
from brevitas.graph import TorchFunctionalToModule
from brevitas.nn import QuantScaledDotProductAttention
from brevitas.nn import ScaledDotProductAttention
from brevitas_examples.llm.llm_quant.mha_layers import QuantizableOPTAttention

QUANTIZABLE_MHA_MAP = {
    OPTAttention: (QuantizableOPTAttention, {
        'batch_first': True}),}

if version.parse(transformers.__version__) >= version.parse('4.46.0'):
    from transformers.models.opt.modeling_opt import OPTSdpaAttention
    QUANTIZABLE_MHA_MAP[OPTSdpaAttention] = (QuantizableOPTAttention, {'batch_first': True})


def replace_mha_with_quantizable_layers(model, dtype):
    rewriters = []
    for src_module, (quantizable_module, quantizable_module_kwargs) in QUANTIZABLE_MHA_MAP.items():
        rewriter = ModuleToModuleByClass(
            src_module, quantizable_module, **quantizable_module_kwargs, dtype=dtype)
        rewriters.append(rewriter)
    if not rewriters:
        warnings.warn(
            f"No module to replace was found. Supported modules are {list(QUANTIZABLE_MHA_MAP.keys())}"
        )
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model


def replace_sdpa_with_quantizable_layers(graph_model):
    fn_to_module_map = ((F.scaled_dot_product_attention, ScaledDotProductAttention),)
    graph_model = TorchFunctionalToModule(fn_to_module_map=fn_to_module_map).apply(graph_model)
    return graph_model


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
