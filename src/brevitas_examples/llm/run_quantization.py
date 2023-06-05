import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers import pipeline

from brevitas_examples.llm.llm_quant.bias_corr import apply_bias_correction
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration
from brevitas_examples.llm.llm_quant.data import get_c4
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization
from brevitas_examples.llm.llm_quant.eval import model_eval
from brevitas_examples.llm.llm_quant.gptq import apply_gptq
from brevitas_examples.llm.llm_quant.quantize import quantize_model
from brevitas_examples.llm.llm_quant.run_utils import get_model_impl


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="facebook/opt-125m", help='HF model name.')
    parser.add_argument(
        '--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument(
        '--nsamples', type=int, default=128, help='Number of calibration data samples.')
    parser.add_argument('--seqlen', type=int, default=2048, help='Sequence length.')
    parser.add_argument('--weight-bit-width', type=int, default=8, help='Weight bit width.')
    parser.add_argument(
        '--weight-param-method',
        type=str,
        default='stats',
        choices=['stats', 'mse'],
        help='How scales/zero-point are determined.')
    parser.add_argument(
        '--weight-scale-type',
        type=str,
        default='float32',
        choices=['float32', 'po2'],
        help='Whether scale is a float value or a po2.')
    parser.add_argument(
        '--weight-quant-type',
        type=str,
        default='asym',
        choices=['sym', 'asym'],
        help='Weight quantization type.')
    parser.add_argument(
        '--weight-quant-granularity',
        type=str,
        default='per_block',
        choices=['per_channel', 'per_tensor', 'per_block'],
        help='Granularity for scales/zero-point of weights.')
    parser.add_argument(
        '--input-bit-width',
        type=int,
        default=None,
        help='Input bit width. Default: None (disabled).')
    parser.add_argument(
        '--input-param-method',
        type=str,
        default='stats',
        choices=['stats', 'mse'],
        help='How scales/zero-point are determined.')
    parser.add_argument(
        '--input-scale-type',
        type=str,
        default='float32',
        choices=['float32', 'po2'],
        help='Whether scale is a float value or a po2.')
    parser.add_argument(
        '--input-quant-type',
        type=str,
        default='asym',
        choices=['sym', 'asym'],
        help='Weight quantization type.')
    parser.add_argument(
        '--input-quant-granularity',
        type=str,
        default='per_tensor',
        choices=['per_tensor'],
        help='Granularity for scales/zero-point of weights.')
    parser.add_argument(
        '--weight-block-size', type=int, default=128, help='Block size for block granularity.')
    parser.add_argument('--gptq', action='store_true', help='Apply GPTQ.')
    parser.add_argument(
        '--act-calibration', action='store_true', help='Apply activation calibration.')
    parser.add_argument('--bias-corr', action='store_true', help='Apply bias correction.')
    parser.add_argument(
        '--act-equalization',
        action='store_true',
        help='Apply activation equalization (SmoothQuant).')

    args = parser.parse_args()
    set_seed(args.seed)

    kwargs = {"torch_dtype": torch.float}
    print("Model loading...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    print("Model loaded.")
    model.eval()

    print("Data loading...")
    calibration_loader, val_data = get_c4(
        nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=args.seqlen)
    print("Data loaded.")

    if args.act_equalization:
        print("Apply act equalization (SmoothQuant)...")
        apply_act_equalization(model, calibration_loader, args.nsamples)
        print("Act equalization applied.")

    print("Applying model quantization...")
    quantize_model(
        get_model_impl(model).layers,
        weight_quant_type=args.weight_quant_type,
        weight_bit_width=args.weight_bit_width,
        weight_param_method=args.weight_param_method,
        weight_scale_type=args.weight_scale_type,
        weight_quant_granularity=args.weight_quant_granularity,
        weight_block_size=args.weight_block_size,
        input_bit_width=args.input_bit_width,
        input_quant_type=args.input_quant_type,
        input_param_method=args.input_param_method,
        input_scale_type=args.input_scale_type,
        input_quant_granularity=args.input_quant_granularity)
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

    print("Model eval...")
    ppl = model_eval(model, val_data, args.seqlen)
    print(f"C4 perplexity: {ppl}")


if __name__ == '__main__':
    main()
