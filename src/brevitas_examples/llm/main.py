"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""

import argparse
import warnings

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from brevitas.export import export_onnx_qcdq
from brevitas.export import export_torch_qcdq
from brevitas_examples.llm.llm_quant.bias_corr import apply_bias_correction
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration
from brevitas_examples.llm.llm_quant.data import get_c4
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization
from brevitas_examples.llm.llm_quant.equalize import apply_weight_equalization
from brevitas_examples.llm.llm_quant.eval import model_eval
from brevitas_examples.llm.llm_quant.gptq import apply_gptq
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_affine_merge
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_mha_with_quantizable_layers
from brevitas_examples.llm.llm_quant.quantize import quantize_model
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
    '--weight-scale-type',
    type=str,
    default='float',
    choices=['float', 'po2'],
    help='Whether scale is a float value or a po2. Default: po2.')
parser.add_argument(
    '--weight-quant-type',
    type=str,
    default='asym',
    choices=['sym', 'asym'],
    help='Weight quantization type. Default: asym.')
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
    '--input-param-method',
    type=str,
    default='stats',
    choices=['stats', 'mse'],
    help='How scales/zero-point are determined. Default: stats.')
parser.add_argument(
    '--input-scale-type',
    type=str,
    default='float',
    choices=['float', 'po2'],
    help='Whether input scale is a float value or a po2. Default: float.')
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
    choices=['per_tensor', 'per_row'],
    help='Granularity for scales/zero-point of inputs. Default: per_tensor.')
parser.add_argument(
    '--quantize-input-zero-point', action='store_true', help='Quantize input zero-point.')
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
    '--act-equalization', action='store_true', help='Apply activation equalization (SmoothQuant).')
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
            if args.input_quant_type == 'asym':
                assert args.quantize_input_zero_point, "Quantized input zero point required."
        if args.export_target == 'torch_qcdq':
            assert args.weight_quant_granularity != 'per_group', "TorchScript QCDQ export doesn't support group weight quantization."
            if args.weight_quant_type == 'asym':
                assert args.quantize_weight_zero_point, "Quantized weight zero point required."
            if args.input_quant_type == 'asym':
                assert args.quantize_input_zero_point, "Quantized input zero point required."
        if args.input_bit_width is not None and not args.act_calibration:
            warnings.warn(
                "Input quantization is being applied without activation calibration. Set --act-calibration."
            )


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
        apply_layernorm_affine_merge(model, ref_kwargs={'input_ids': calibration_loader[0]})
        print("LN affine merge applied.")

    # Insert standard MHA layers when performing weight equalization to avoid dealing
    # with all the variability in HF implementations
    if args.weight_equalization or args.input_bit_width:
        print("Replace HF MHA with quantizable variants...")
        model = replace_mha_with_quantizable_layers(model, dtype)
        print("Replacing done.")

    if args.weight_equalization:
        print("Apply weight equalization...")
        apply_weight_equalization(model, ref_kwargs={'input_ids': calibration_loader[0]})
        print("Weight equalization applied.")

    if args.act_equalization:
        print("Apply act equalization (SmoothQuant)...")
        apply_act_equalization(model, calibration_loader, args.nsamples)
        print("Act equalization applied.")

    if not args.no_quantize:
        print("Applying model quantization...")
        quantize_model(
            get_model_impl(model).layers,
            dtype=dtype,
            weight_quant_type=args.weight_quant_type,
            weight_bit_width=args.weight_bit_width,
            weight_param_method=args.weight_param_method,
            weight_scale_type=args.weight_scale_type,
            weight_quant_granularity=args.weight_quant_granularity,
            weight_group_size=args.weight_group_size,
            quantize_weight_zero_point=args.quantize_weight_zero_point,
            input_bit_width=args.input_bit_width,
            input_quant_type=args.input_quant_type,
            input_param_method=args.input_param_method,
            input_scale_type=args.input_scale_type,
            input_quant_granularity=args.input_quant_granularity,
            quantize_input_zero_point=args.quantize_input_zero_point,
            seqlen=args.seqlen)
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
        model_export(model, calibration_loader[0], args)


if __name__ == '__main__':
    main()
