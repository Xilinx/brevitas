import argparse

import torch
from torch import nn
import torch_mlir

from brevitas.backport.fx._symbolic_trace import wrap
from brevitas.backport.fx.experimental.proxy_tensor import make_fx
from brevitas_examples.llm.llm_quant.export import block_quant_layer_level_manager
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.llm.llm_quant.export import brevitas_layer_export_mode
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from brevitas_examples.llm.llm_quant.export import LinearWeightBlockQuantHandler
from brevitas_examples.llm.llm_quant.export import replace_call_fn_target
from brevitas_examples.llm.llm_quant.mlir_custom_mm import brevitas_matmul_rhs_group_quant_library
from brevitas_examples.llm.llm_quant.quantize import quantize_model


# Due a tracing issue this annotation needs to be
# in the same module (== file) from which make_fx is called
# We also can't directly annotate torch.ops.quant.matmul_rhs_group_quant
# and so we trace a placeholder first and then replace it post tracing
@wrap(visible_to_make_fx=True)
def matmul_rhs_group_quant_placeholder(*args, **kwargs):
    return torch.ops.quant.matmul_rhs_group_quant(*args, **kwargs)


class LinearWeightBlockQuantHandlerFwd(LinearWeightBlockQuantHandler):

    def forward(self, x):
        # Due a tracing issue the call to this fn needs to be
        # in the same module (== file) from which make_fx is called
        out = matmul_rhs_group_quant_placeholder(
            x, self.int_weight, self.scale, self.zero_point, self.bit_width, self.group_size)
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(128, 256, bias=True)

    def forward(self, x):
        return self.layer(x)


def quantize_and_export(args):
    # Init model
    model = Model()

    # Run quantization
    quantize_model(
        model,
        dtype=torch.float32,
        weight_quant_type=args.weight_quant_type,
        weight_bit_width=args.weight_bit_width,
        weight_group_size=args.weight_group_size,
        weight_param_method='stats',
        weight_scale_precision='float',
        weight_quant_granularity='per_group',
        quantize_weight_zero_point=False)

    # Run a test forward pass
    model(torch.randn(2, 128))

    # Pick export mode
    if not args.no_custom_packed_export:
        export_context_manager = brevitas_layer_export_mode
        # we generate an export_class since we need to pass in the handler defined above
        export_class = block_quant_layer_level_manager(
            export_handlers=[LinearWeightBlockQuantHandlerFwd])
    else:
        export_context_manager = brevitas_proxy_export_mode
        export_class = BlockQuantProxyLevelManager

    # export with make_fx with support for fx wrap
    with export_context_manager(model, export_class):
        traced_model = make_fx(model)(torch.randn(2, 128))

    # Replace placeholder for custom op with correct call, if any
    replace_call_fn_target(
        traced_model,
        src=matmul_rhs_group_quant_placeholder,
        target=torch.ops.quant.matmul_rhs_group_quant)

    # print the output graph
    print(traced_model.graph)

    torch_mlir.compile(
        traced_model,
        torch.randn(2, 128),
        output_type="torch",
        backend_legal_ops=["quant.matmul_rhs_group_quant"],
        extra_library=brevitas_matmul_rhs_group_quant_library,
        use_tracing=True,
        verbose=False)


def main():
    parser = argparse.ArgumentParser(
        description='Export single linear with weight group quant to torch-mlir.')
    parser.add_argument('--weight-bit-width', type=int, default=8, help='Weight bit width.')
    parser.add_argument(
        '--weight-quant-type',
        type=str,
        default='asym',
        choices=['sym', 'asym'],
        help='Weight quantization type.')
    parser.add_argument(
        '--weight-group-size',
        type=int,
        default=128,
        help='Group size for group weight quantization.')
    parser.add_argument(
        '--no-custom-packed-export',
        action='store_true',
        help='Enable export to a custom mm op with packed weights for int2 and int4.')
    args = parser.parse_args()
    quantize_and_export(args)


if __name__ == "__main__":
    main()
