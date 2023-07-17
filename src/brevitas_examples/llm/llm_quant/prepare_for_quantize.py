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
    for rewriter in rewriters:
        model = rewriter.apply(model)
    return model
