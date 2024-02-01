from argparse import ArgumentParser
import os

import torch

from brevitas_examples.llm.llm_quant.data import get_wikitext2

torch.manual_seed(0)
import warnings

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.utils.fx import symbolic_trace

from brevitas_examples.llm.llm_quant.eval import model_eval_accelerate
from brevitas_examples.optimum.quantizer import BrevitasQuantizationConfig
from brevitas_examples.optimum.quantizer import BrevitasQuantizer
from brevitas_examples.optimum.utils import offload_model
from brevitas_examples.optimum.utils import remove_hooks

warnings.warn((
    "To run this example, it is required to install accelerate from source, "
    "e.g., pip install git+https://github.com/huggingface/accelerate.git@main"))

warnings.warn((
    "To run this example, it is required to install transformers from source, "
    "e.g., pip install git+https://github.com/huggingface/transformers.git@main"))

warnings.warn((
    "To run this example, it is required to install optimum from source, "
    "e.g., pip install git+https://github.com/huggingface/optimum.git@main"))

parser = ArgumentParser(
    description="A simple example to demonstrate a prototype Brevitas/HuggingFace quantization flow"
)
parser.add_argument(
    "--apply-gptq",
    action="store_true",
    default=False,
    help="Apply the GPTQ algorithm during quantization (Note, currently slow! default: %(default)s)"
)
parser.add_argument(
    "--apply-weight-equalization",
    action="store_true",
    default=False,
    help="Apply the weight equalization algorithm (default: %(default)s)")
parser.add_argument(
    "--apply-act-equalization",
    type=str,
    choices=["fx", "layerwise"],
    default=None,
    help=
    "Apply the activation equalization (SmoothQuant) algorithm (choices: [%(choices)s, None], default: %(default)s)"
)
parser.add_argument(
    "--replace-mha-with-quantizable",
    action="store_true",
    default=False,
    help="Replace attention with standard PyTorch implementation (default: %(default)s)")
parser.add_argument(
    "--with-fx",
    action="store_true",
    default=False,
    help="Convert to an FX GraphModule before applying quantization (default: %(default)s)")
args = parser.parse_args()

model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

model.eval()

via_fx = args.with_fx or args.apply_act_equalization == "fx"
if via_fx:
    dtype = next(iter(model.parameters())).dtype
    input_names = ["input_ids", "attention_mask", "past_key_values"]

    # Determine past_key_values tuple shapes
    num_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    embed_dim = model.config.hidden_size // num_attention_heads
    with torch.no_grad():
        model = symbolic_trace(model, input_names)

    pkv = tuple((
        torch.empty(1, num_attention_heads, 0, embed_dim, dtype=dtype),
        torch.empty(1, num_attention_heads, 0, embed_dim, dtype=dtype)) for i in range(num_layers))

    def forward_call(model, inps):
        inps['past_key_values'] = pkv

        return model(**inps)
else:

    def forward_call(model, inps):
        return model(**inps)


config = BrevitasQuantizationConfig(
    apply_gptq=args.apply_gptq,
    apply_weight_equalization=args.apply_weight_equalization,
    apply_act_equalization=args.apply_act_equalization,
    replace_mha_with_quantizable=args.replace_mha_with_quantizable)
quantizer = BrevitasQuantizer(model, config)

# To speed up GPTQ computation, we can look through the model to find layers that can be optimized in parallel
# Because they do not depend on each other. A typical case is the input matrices of the attention layer.
# We just need to specify the suffix of the layer, and they will be matched across the entire structure.
# quantizer.find_groups_of_parallel_layers([['q_proj', 'k_proj', 'v_proj']]) # This is for base OPT

calibration_dataloader = quantizer.get_calibration_dataloader(
    tokenizer, dataset_name='wikitext2-raw', num_samples=config.nsamples, seqlen=config.seqlen)

print("Apply quantization")
model = quantizer.quantize(model, calibration_dataloader, forward_call=forward_call)
print("Quantization applied")

model = offload_model(model)
print("Model eval...")
validation_dataloader = get_wikitext2(
    config.nsamples, 0, config.seqlen, tokenizer, type='raw', split='validation')
ppl = model_eval_accelerate(model, validation_dataloader, config.seqlen, forward_call=forward_call)
print(f"C4 perplexity: {ppl}")

print("Model export...")
# Several export format could be achieved:
# QCDQ, Packed weight + QCDQ for activation, Packed weights + Dynamic Quant for activations...
remove_hooks(model)

## Export still WIP
quantizer.export(model, 'model_export')
