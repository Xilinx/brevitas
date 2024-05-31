import warnings

from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.opt.modeling_opt import OPTAttention

from brevitas.graph import ModuleToModuleByClass
from brevitas_examples.llm.llm_quant.mha_layers import QuantizableBertAttention
from brevitas_examples.llm.llm_quant.mha_layers import QuantizableOPTAttention

QUANTIZABLE_MHA_MAP = {
    OPTAttention: (QuantizableOPTAttention, {
        'batch_first': True}),
    BertAttention: (QuantizableBertAttention, {
        'batch_first': True}),}


def _set_bert_mha_attributes(module):
    module.all_head_size = module._modules['self'].all_head_size
    module.num_attention_heads = module._modules['self'].num_attention_heads
    module.ln_normalized_shape = module._modules['output'].LayerNorm.normalized_shape
    module.ln_eps = module._modules['output'].LayerNorm.eps
    module.ln_elementwise_affine = module._modules['output'].LayerNorm.elementwise_affine
    module.ln_bias = False if module._modules['output'].LayerNorm.bias is None else True


_SET_ATTRIBUTES_MAP = {
    BertAttention: _set_bert_mha_attributes,
}


def set_mha_attributes(model):
    for name, module in model.named_modules():
        mod_type = type(module)
        if mod_type in _SET_ATTRIBUTES_MAP.keys():
            _SET_ATTRIBUTES_MAP[mod_type](module)
    return model


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
