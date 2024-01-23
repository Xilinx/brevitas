from argparse import ArgumentParser
import os

import torch

torch.manual_seed(0)
import warnings

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from brevitas_examples.llm.llm_quant.eval import model_eval_accelerate
from brevitas_examples.optimum.quantizer import BrevitasQuantizationConfig
from brevitas_examples.optimum.quantizer import BrevitasQuantizer
from brevitas_examples.optimum.utils import offload_model
from brevitas_examples.optimum.utils import remove_hooks

warnings.warn((
    "To run this example, it is required to install accelerate from source, "
    "e.g., pip install git+https://github.com/huggingface/accelerate.git@main"))

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
input_names = ["input_ids", "attention_mask", "past_key_values"]


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["sentence"], padding="max_length", max_length=128, truncation=True)


config = BrevitasQuantizationConfig(
    apply_gptq=args.apply_gptq,
    apply_weight_equalization=args.apply_weight_equalization,
    apply_act_equalization='layerwise',  #args.apply_act_equalization,
    replace_mha_with_quantizable=args.replace_mha_with_quantizable,
)
via_fx = args.with_fx or args.apply_act_equalization == "fx"
quantizer = BrevitasQuantizer(model, config)

calibration_dataloader = quantizer.get_calibration_dataloader(
    tokenizer, dataset_name='wikitext2-raw', num_samples=config.nsamples, seqlen=config.seqlen)
if via_fx:
    model = get_fx_graph(
        model,
        ref_kwargs={
            'input_ids': calibration_dataloader[0]
        },  # , 'attention_mask': torch.ones_like(calibration_dataloader[0])}, adding this will make the attention_mask a required argument, leaving it out removes it from the forward signature
        dtype=torch.float32)

from brevitas_examples.llm.llm_quant.run_utils import get_fx_graph

model = quantizer.quantize(model, calibration_dataloader)

model = offload_model(model)
print("Model eval...")
validation_dataloader = traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
validation_dataloader = tokenizer("\n\n".join(validation_dataloader['text']), return_tensors='pt')
ppl = model_eval_accelerate(model, validation_dataloader, config.seqlen)
print(f"C4 perplexity: {ppl}")

print("Model export...")
# Several export format could be achieved:
# QCDQ, Packed weight + QCDQ for activation, Packed weights + Dynamic Quant for activations...
remove_hooks(model)
quantizer.export(model, 'model_export.onnx')
