"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import argparse
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from brevitas.export import export_onnx_qcdq
from brevitas.export import export_torch_qcdq
from brevitas_examples.common.generative.quantize import quantize_model
from brevitas_examples.common.parse_utils import quant_format_validator
from brevitas_examples.llm.llm_quant.bias_corr import apply_bias_correction
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration
from brevitas_examples.llm.llm_quant.data import get_c4
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization
from brevitas_examples.llm.llm_quant.equalize import apply_weight_equalization
from brevitas_examples.llm.llm_quant.eval import model_eval
from brevitas_examples.llm.llm_quant.gptq import apply_gptq
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_affine_merge
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_mha_with_quantizable_layers
from brevitas_examples.llm.llm_quant.run_utils import CastFloat16ToFloat32
from brevitas_examples.llm.llm_quant.run_utils import get_model_impl

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    type=str,
    default="facebook/opt-125m",
    help='HF model name. Default: facebook/opt-125m.')
parser.add_argument(
    '--seed', type=int, default=0, help='Seed for sampling the calibration data. Default: 0.')
parser.add_argument(
    '--nsamples', type=int, default=128, help='Number of calibration data samples. Default: 128.')
parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length. Default: 2048.')
parser.add_argument('--eval', action='store_true', help='Eval model PPL on C4.')
parser.add_argument('--weight-bit-width', type=int, default=8, help='Weight bit width. Default: 8.')
parser.add_argument(
    '--weight-param-method',
    type=str,
    default='stats',
    choices=['stats', 'mse'],
    help='How scales/zero-point are determined. Default: stats.')
parser.add_argument(
    '--weight-scale-precision',
    type=str,
    default='float_scale',
    choices=['float_scale', 'po2_scale'],
    help='Whether scale is a float value or a po2. Default: po2.')
parser.add_argument(
    '--weight-quant-type',
    type=str,
    default='asym',
    choices=['sym', 'asym'],
    help='Weight quantization type. Default: asym.')
parser.add_argument(
    '--weight-quant-format',
    type=quant_format_validator,
    default='int',
    help='Weight quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. Default: int.'
)
parser.add_argument(
    '--weight-quant-granularity',
    type=str,
    default='per_group',
    choices=['per_channel', 'per_tensor', 'per_group'],
    help='Granularity for scales/zero-point of weights. Default: per_group.')
parser.add_argument(
    '--weight-group-size',
    type=int,
    default=128,
    help='Group size for per_group weight quantization. Default: 128.')
parser.add_argument(
    '--quantize-weight-zero-point', action='store_true', help='Quantize weight zero-point.')
parser.add_argument(
    '--input-bit-width',
    type=int,
    default=None,
    help='Input bit width. Default: None (disables input quantization).')
parser.add_argument(
    '--input-quant-format',
    type=quant_format_validator,
    default='int',
    help='Input quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. Default: int.')
parser.add_argument(
    '--input-param-method',
    type=str,
    default='stats',
    choices=['stats', 'mse'],
    help=
    'How scales/zero-point are determined. Default: stats (percentile for static, absmax or minmax for dynamic).'
)
parser.add_argument(
    '--input-scale-precision',
    type=str,
    default='float_scale',
    choices=['float_scale', 'po2_scale'],
    help='Whether input scale is a float value or a po2. Default: float.')
parser.add_argument(
    '--input-scale-type',
    type=str,
    default='static',
    choices=['static', 'dynamic', 'no_scale'],
    help='Whether input scale is a static value or a dynamic value.')
parser.add_argument(
    '--input-quant-type',
    type=str,
    default='asym',
    choices=['sym', 'asym'],
    help='Input quantization type. Default: asym.')
parser.add_argument(
    '--input-quant-granularity',
    type=str,
    default='per_tensor',
    choices=['per_tensor', 'per_row', 'per_group'],
    help='Granularity for scales/zero-point of inputs. Default: per_tensor.')
parser.add_argument(
    '--input-group-size',
    type=int,
    default=64,
    help='Group size for per_group input quantization. Default: 64.')
parser.add_argument(
    '--quantize-input-zero-point', action='store_true', help='Quantize input zero-point.')
parser.add_argument(
    '--quantize-embedding', action='store_true', help='Quantize first nn.Embedding layer.')
parser.add_argument(
    '--quantize-last-layer', action='store_true', help='Quantize last nn.Linear layer.')
parser.add_argument('--gptq', action='store_true', help='Apply GPTQ.')
parser.add_argument('--act-calibration', action='store_true', help='Apply activation calibration.')
parser.add_argument('--bias-corr', action='store_true', help='Apply bias correction.')
parser.add_argument('--ln-affine-merge', action='store_true', help='Merge LN affine params.')
parser.add_argument('--no-quantize', action='store_true', help='Disable quantization.')
parser.add_argument(
    '--no-float16',
    action='store_true',
    help='Disable float16 as base datatype and switch to float32.')
parser.add_argument(
    '--weight-equalization',
    action='store_true',
    help='Apply weight equalization. Relevant to ReLU based models (e.g. OPT).')
parser.add_argument(
    '--act-equalization',
    default=None,
    choices=[None, 'layerwise', 'fx'],
    help='Apply activation equalization (SmoothQuant). Layerwise introduces standalone mul nodes,'
    'while fx merges them whenever possible into previous tensors, which is possible on ReLU based models (e.g. OPT).'
)
parser.add_argument('--load-awq', type=str, default=None, help="Load the awq search results.")
parser.add_argument(
    '--export-target',
    default=None,
    choices=[
        None,
        'onnx_qcdq',
        'torch_qcdq',
        'sharded_torchmlir_group_weight',
        'sharded_packed_torchmlir_group_weight'],
    help='Model export.')


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def model_export(model, ref_input, args):
    if args.export_target == 'sharded_torchmlir_group_weight':
        from brevitas_examples.llm.llm_quant.sharded_mlir_group_export import \
            sharded_weight_group_export
        sharded_weight_group_export(model, no_custom_packed_export=True)
    elif args.export_target == 'sharded_packed_torchmlir_group_weight':
        from brevitas_examples.llm.llm_quant.sharded_mlir_group_export import \
            sharded_weight_group_export
        sharded_weight_group_export(model, no_custom_packed_export=False)
    elif args.export_target == 'onnx_qcdq':
        export_onnx_qcdq(model, ref_input, export_path=f"{args.model.replace('/', '-')}.onnx")
    elif args.export_target == 'torch_qcdq':
        export_torch_qcdq(model, ref_input, export_path=f"{args.model.replace('/', '-')}.pt")


def validate(args):
    if not args.no_quantize:
        if args.export_target is not None:
            assert args.input_quant_format == 'int', "Only integer quantization supported for export currently."
        if args.export_target is not None and args.input_bit_width is not None:
            assert args.input_scale_type == 'static', "Only static scale supported for export currently."
        if args.export_target == 'sharded_torchmlir_group_weight':
            assert args.weight_quant_granularity == 'per_group', "Sharded torch group export requires per group weight quant."
            assert args.input_bit_width is None, "Sharded torch group weight export doesn't support input quant."
            assert not args.quantize_weight_zero_point, "Quantized weight zero point not supported."
        if args.export_target == 'sharded_packed_torchmlir_group_weight':
            assert args.weight_quant_granularity == 'per_group', "Sharded torch group export requires per group weight quant."
            assert args.input_bit_width is None, "Sharded packed torch group weight export doesn't support input quant."
            assert not args.quantize_weight_zero_point, "Quantized weight zero point not supported."
        if args.export_target == 'onnx_qcdq':
            assert args.weight_quant_granularity != 'per_group', "ONNX QCDQ export doesn't support group weight quantization."
            if args.weight_quant_type == 'asym':
                assert args.quantize_weight_zero_point, "Quantized weight zero point required."
            if args.input_bit_width is not None and args.input_quant_type == 'asym':
                assert args.quantize_input_zero_point, "Quantized input zero point required."
        if args.export_target == 'torch_qcdq':
            assert args.weight_quant_granularity != 'per_group', "TorchScript QCDQ export doesn't support group weight quantization."
            if args.weight_quant_type == 'asym':
                assert args.quantize_weight_zero_point, "Quantized weight zero point required."
            if args.input_bit_width is not None and args.input_quant_type == 'asym':
                assert args.quantize_input_zero_point, "Quantized input zero point required."
        if (args.input_bit_width and
            (args.input_scale_type == 'static' or
             (args.input_scale_type == 'dynamic' and args.input_quant_type == 'asym'))):
            assert args.act_calibration, "Static input quantization is being applied without activation calibration. Set --act-calibration."


def main():
    args = parser.parse_args()
    validate(args)
    set_seed(args.seed)

    if args.no_float16:
        dtype = torch.float32
    else:
        dtype = torch.float16

    kwargs = {"torch_dtype": dtype}
    print("Model loading...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    print("Model loaded.")
    model.eval()

    if args.load_awq:
        from brevitas_examples.llm.llm_quant.awq.pre_quant import apply_awq
        awq_results = torch.load(args.load_awq, map_location="cpu")
        with CastFloat16ToFloat32():
            apply_awq(model, awq_results)

    if (args.export_target or args.eval or args.act_equalization or args.act_calibration or
            args.gptq or args.bias_corr or args.ln_affine_merge or args.weight_equalization):
        print("Data loading...")
        calibration_loader, val_data = get_c4(
            nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=args.seqlen)
        print("Data loaded.")

    # Apply LN affine merging before inserting MHA layers
    # since currently there is support only for merging into Linear
    if args.ln_affine_merge:
        print("Apply LN affine merge...")
        apply_layernorm_affine_merge(model, dtype, ref_kwargs={'input_ids': calibration_loader[0]})
        print("LN affine merge applied.")

    # Insert standard MHA layers when performing fx based weight/act equalization to avoid dealing
    # with all the variability in HF implementations
    if args.weight_equalization or args.act_equalization == 'fx' or args.input_bit_width:
        print("Replace HF MHA with quantizable variants...")
        model = replace_mha_with_quantizable_layers(model, dtype)
        print("Replacing done.")

    if args.weight_equalization:
        print("Apply weight equalization...")
        apply_weight_equalization(model, dtype, ref_kwargs={'input_ids': calibration_loader[0]})
        print("Weight equalization applied.")

    if args.act_equalization is not None:
        print("Apply act equalization (SmoothQuant)...")
        apply_act_equalization(
            model,
            dtype,
            args.act_equalization,
            calibration_loader,
            args.nsamples,
            ref_kwargs={'input_ids': calibration_loader[0]})
        print("Act equalization applied.")

    if args.quantize_embedding or args.quantize_last_layer:
        layers_to_quantize = model
    else:
        layers_to_quantize = get_model_impl(model).layers

    if not args.no_quantize:
        print("Applying model quantization...")
        quantize_model(
            layers_to_quantize,
            dtype=dtype,
            weight_quant_format=args.weight_quant_format,
            weight_quant_type=args.weight_quant_type,
            weight_bit_width=args.weight_bit_width,
            weight_param_method=args.weight_param_method,
            weight_scale_precision=args.weight_scale_precision,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            quantize_weight_zero_point=args.quantize_weight_zero_point,
            input_bit_width=args.input_bit_width,
            input_quant_type=args.input_quant_type,
            input_quant_format=args.input_quant_format,
            input_param_method=args.input_param_method,
            input_scale_precision=args.input_scale_precision,
            input_scale_type=args.input_scale_type,
            input_quant_granularity=args.input_quant_granularity,
            input_group_size=args.input_group_size,
            quantize_input_zero_point=args.quantize_input_zero_point,
            quantize_embedding=args.quantize_embedding,
            seqlen=args.seqlen)
        # Tie back first/last layer weights in case they got untied
        model.tie_weights()
        print(model)
        print("Model quantization applied.")

    if args.act_calibration:
        print("Apply act calibration...")
        apply_calibration(model, calibration_loader, args.nsamples)
        print("Act calibration applied.")

    if args.gptq:
        print("Applying GPTQ...")
        apply_gptq(model, calibration_loader, args.nsamples)
        print("GPTQ applied.")

    if args.bias_corr:
        print("Applying bias correction...")
        apply_bias_correction(model, calibration_loader, args.nsamples)
        print("Bias correction applied.")

    if args.eval:
        print("Model eval...")
        ppl = model_eval(model, val_data, args.seqlen)
        print(f"C4 perplexity: {ppl}")

    if args.export_target:
        print(f"Export to {args.export_target}")
        # Currently we always export on CPU with a float32 container to avoid float16 CPU errors
        model = model.cpu().to(dtype=torch.float32)
        model_export(model, calibration_loader[0], args)


if __name__ == '__main__':
    main()
