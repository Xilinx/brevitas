
import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
import functools
import pprint
import sys

from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPActPerTensorFloat, Fp8e4m3OCPWeightPerTensorFloat
import numpy as np
from optimum.exporters.onnx import onnx_export_from_model
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import yaml
import brevitas.nn as qnn
import torch.nn as nn

from brevitas.export.inference.manager import quant_inference_mode
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas.graph import load_quant_model_mode
from brevitas.graph.base import ModuleInstanceTransformTensor
from brevitas.graph.equalize import fuse_parametrizations
from brevitas.graph.equalize import GraphRotationEqualization
from brevitas.graph.equalize import LayerwiseActivationRotation
from brevitas.graph.quantize import functional_quantization_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.utils import get_module
from brevitas.graph.utils import remove_weight_orig
from brevitas.nn.quant_sdpa import ScaledDotProductAttention
from brevitas.utils.python_utils import hooked_on_a_function
from brevitas_examples.common.accelerate_utils.accelerate import offload_model
from brevitas_examples.common.accelerate_utils.accelerate import remove_hooks
from brevitas_examples.common.accelerate_utils.accelerate import update_internal_dict
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.llm.gguf_export.export import save_quantized_as_gguf
from brevitas_examples.llm.llm_args import create_llm_args_parser
from brevitas_examples.llm.llm_args import validate
from brevitas_examples.llm.llm_quant.awq.pre_quant import apply_awq
from brevitas_examples.llm.llm_quant.bias_corr import apply_bias_correction
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration
from brevitas_examples.llm.llm_quant.data_utils import get_dataset_for_model
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization
from brevitas_examples.llm.llm_quant.equalize import apply_weight_equalization
from brevitas_examples.llm.llm_quant.eval import compute_perplexity
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from brevitas_examples.llm.llm_quant.gpxq import apply_gpfq
from brevitas_examples.llm.llm_quant.gpxq import apply_gptq
from brevitas_examples.llm.llm_quant.gpxq import apply_magr
from brevitas_examples.llm.llm_quant.gpxq import apply_qronos
from brevitas_examples.llm.llm_quant.learned_round_utils import apply_learned_round
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_affine_merge
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_to_rmsnorm
from brevitas_examples.llm.llm_quant.ln_affine_merge import replace_rmsnorm_with_torch
from brevitas_examples.llm.llm_quant.prepare_for_quantize import add_zero_bias_to_linear, replace_mlperf_attn
from brevitas_examples.llm.llm_quant.prepare_for_quantize import \
    replace_sdpa_with_quantizable_layers
from brevitas_examples.llm.llm_quant.rotation_optimization import apply_rotation_optimization
from brevitas_examples.llm.llm_quant.rotation_optimization import parse_rotation_optimization_args
from brevitas_examples.llm.llm_quant.run_utils import fix_rewriter
from brevitas_examples.llm.llm_quant.svd_quant import apply_svd_quant
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3Act
from brevitas.quant.experimental.mx_quant_ocp import MXFloat8e4m3WeightMSE
from brevitas.core.function_wrapper import FloorSte


class MXFP4Weight(MXFloat8e4m3WeightMSE):
    restrict_value_float_to_int_impl = FloorSte

class MXFP4Act(MXFloat8e4m3Act):
    restrict_value_float_to_int_impl = FloorSte

class FP8Weight(Fp8e4m3OCPWeightPerTensorFloat):
    pass

class FP8Act(Fp8e4m3OCPActPerTensorFloat):
    pass


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def main(args):
    set_seed(0)

    # Whether to quantize SDPA with FX

    kwargs = {"torch_dtype": args.dtype}
    if args.quant_sdpa:
        kwargs["attn_implementation"] = "sdpa"

    print("Model loading...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    dtype = next(model.parameters()).dtype
    print("Model loaded.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load the data for calibration and evaluation.
    calibration_loader = get_dataset_for_model(
        args.model,
        bos_preprocessing=args.bos_preprocessing,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="train",
        seed=args.seed,
        require_fx=False,
        device=None)

    validation_loader = get_dataset_for_model(
        args.model,
        bos_preprocessing=args.bos_preprocessing,
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        split="validation",
        seed=args.seed,
        require_fx=False,
        device=None)
    

    device = next(iter(model.parameters())).device
    print("Data loaded.")

    print("Applying MagR...")
    model = offload_model(model)
    apply_magr(
        model,
        calibration_loader,
        create_weight_orig=not args.disable_create_weight_orig,
        alpha=args.magr_alpha)
    remove_hooks(model)
    print(f"MagR applied.")

    model = replace_mlperf_attn(model)

    # Attn Quantization
    q_scaled_quant = FP8Act
    k_transposed_quant = FP8Act
    q_scaled_quant = q_scaled_quant.let(
    **{
        'group_dim': -1, 'group_size': 32
    })
    k_transposed_quant = k_transposed_quant.let(
        **{
            'group_dim': -2, 'group_size': 32})
    v_quant = k_transposed_quant

    quant_sdpa_kwargs = {
        'softmax_input_quant': None,
        'attn_output_weights_quant': None,
        'q_scaled_quant': q_scaled_quant,
        'k_transposed_quant': k_transposed_quant,
        'v_quant': v_quant,
        'attn_output_quant': None,
        'dtype': dtype,
        'device': device}
    
    mxfp4_layer_types = []

    quant_linear_kwargs = {
        'input_quant': lambda name, module: MXFP4Act if any([pattern in name for pattern in mxfp4_layer_types]) else FP8Act,
        'weight_quant': lambda name, module: MXFP4Weight if any([pattern in name for pattern in mxfp4_layer_types]) else FP8Weight,
    }


    layer_map = {
        nn.Linear: (qnn.QuantLinear, quant_linear_kwargs),
        qnn.ScaledDotProductAttention: (qnn.QuantScaledDotProductAttention, quant_sdpa_kwargs)}
    name_blacklist = []
    
    model = layerwise_quantize(
        model=model, compute_layer_map=layer_map, name_blacklist=name_blacklist)
    # Just to be sure
    model.eval()
    model = model.to(dtype)

    model = offload_model(model)


    # We initialize weights scale factor
    with torch.no_grad():
        model(**calibration_loader[0])

    if args.compile_ptq:
        for m in model.modules():
            if hasattr(m, 'compile_quant'):
                m.compile_quant()

    if args.act_calibration and not args.load_checkpoint:
        print("Apply act calibration...")
        apply_calibration(model, calibration_loader)
        print("Act calibration applied.")


    if args.gptq and not args.load_checkpoint:
        print("Applying GPTQ...")
        apply_gptq(
            model,
            calibration_loader,
            act_order=True,
            use_quant_activations=True,
            create_weight_orig=False,
            block_name='model.layers')
        print("GPTQ applied.")

    validation_data = []
    for data in validation_data:
        pass
