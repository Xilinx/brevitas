import argparse

import torch
from torch import nn

from brevitas_examples.llm.llm_quant.export import brevitas_layer_export_mode
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from brevitas_examples.llm.llm_quant.make_fx import wrappable_make_fx
from brevitas_examples.llm.llm_quant.quantize import quantize
from brevitas_examples.llm.llm_quant.quantizers import IntWeightSymmetricBlockQuant
from brevitas_examples.llm.llm_quant.quantizers import UintWeightAsymmetricBlockQuant


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 256, bias=True)

    def forward(self, x):
        return self.layer(x)


def quantize_and_export(args):
    # Init model
    model = Model()

    # Pick weight quantizer
    if args.symmetric:
        weight_quant = IntWeightSymmetricBlockQuant
    else:
        weight_quant = UintWeightAsymmetricBlockQuant

    # Run quantization
    quantize(
        model, weight_quant, weight_bit_width=args.bit_width, weight_block_size=args.block_size)

    # Run a test forward pass
    model(torch.randn(2, 128))

    # Pick export mode
    if args.custom_packed_export:
        export_context_manager = brevitas_layer_export_mode
    else:
        export_context_manager = brevitas_proxy_export_mode

    # export with make_fx with support for fx wrap
    with export_context_manager(model):
        traced_model = wrappable_make_fx(model)(torch.randn(2, 128))

    # Output graph
    print(traced_model.graph)


def main():
    parser = argparse.ArgumentParser(description='Llama-based LLM weight only quantization')
    parser.add_argument('-b', '--bit-width', type=int, default=8, help='Weight bit width.')
    parser.add_argument(
        '-s', '--block-size', type=int, default=128, help='Weight quantization block size.')
    parser.add_argument(
        '--symmetric',
        action='store_true',
        help='Enable symmetric weight quantization instead of asymmetric.')
    parser.add_argument(
        '--custom_packed_export',
        action='store_true',
        help='Enable export to a custom mm op with packed weights for int2 and int4.')
    args = parser.parse_args()
    quantize_and_export(args)


if __name__ == "__main__":
    main()
