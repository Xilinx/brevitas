import warnings

import torch
from transformers.models.opt.modeling_opt import OPTAttention

from brevitas.graph import ModuleToModuleByClass
from brevitas_examples.llm.llm_quant.mha_layers import QuantizableOPTAttention

QUANTIZABLE_MHA_MAP = {OPTAttention: (QuantizableOPTAttention, {'batch_first': True})}


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
