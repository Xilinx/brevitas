# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from functools import partial
from itertools import product
import os
import random
from types import SimpleNamespace

import numpy as np
from optimum.amd.brevitas.accelerate_utils import offload_model
from optimum.amd.brevitas.accelerate_utils import remove_hooks
from optimum.amd.brevitas.data_utils import compute_perplexity
from optimum.exporters.onnx import onnx_export_from_model
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from brevitas import __version__ as brevitas_version
from brevitas import config
from brevitas import torch_version
from brevitas.export import export_torch_qcdq
from brevitas.export.onnx.standard.qcdq.manager import StdQCDQONNXManager
from brevitas.graph.quantize import layerwise_quantize
from brevitas_examples.common.generative.quantize import generate_quant_maps
from brevitas_examples.common.generative.quantize import generate_quantizers
from brevitas_examples.common.parse_utils import quant_format_validator
from brevitas_examples.imagenet_classification.ptq.learned_round_utils import \
    apply_learned_round_learning_llm
from brevitas_examples.imagenet_classification.ptq.utils import get_gpu_index
from brevitas_examples.imagenet_classification.ptq.utils import get_next_available_gpu
from brevitas_examples.imagenet_classification.utils import SEED
from brevitas_examples.imagenet_classification.utils import validate
from brevitas_examples.llm.llm_quant.bias_corr import apply_bias_correction
from brevitas_examples.llm.llm_quant.calibrate import apply_calibration
from brevitas_examples.llm.llm_quant.data_utils import get_dataset_for_model
from brevitas_examples.llm.llm_quant.equalize import apply_act_equalization
from brevitas_examples.llm.llm_quant.equalize import apply_weight_equalization
from brevitas_examples.llm.llm_quant.export import BlockQuantProxyLevelManager
from brevitas_examples.llm.llm_quant.export import brevitas_proxy_export_mode
from brevitas_examples.llm.llm_quant.gpxq import apply_gpfq
from brevitas_examples.llm.llm_quant.gpxq import apply_gptq
from brevitas_examples.llm.llm_quant.ln_affine_merge import apply_layernorm_affine_merge
from brevitas_examples.llm.llm_quant.prepare_for_quantize import add_zero_bias_to_linear
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_mha_with_quantizable_layers
from brevitas_examples.llm.llm_quant.run_utils import CastFloat16ToFloat32
from brevitas_examples.llm.llm_quant.run_utils import get_fx

config.IGNORE_MISSING_KEYS = True


def parse_type(v, default_type):
    if v == 'None':
        return None
    else:
        return default_type(v)


def parse_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class hashabledict(dict):

    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]


# Torchvision models with top1 accuracy
LLM_TOP1_MAP = {
    'facebook/opt-125m': None,
    'meta-llama/Llama-2-7b-hf': None,}

OPTIONS_DEFAULT = {
    'model': list(LLM_TOP1_MAP.keys()),  # HF model name. Default: facebook/opt-125m.
    'seed': [0],  # Seed for sampling the calibration data. Default: 0.
    'nsamples': [128],  # Number of calibration data samples. Default: 128.
    'seqlen': [2048],  # Sequence length. Default: 2048.
    'eval': [True],  # Eval model PPL on the chosen Dataset.
    'dataset': ['c4'],  # Dataset to use for quantization (default: wikitext2)
    'weight_bit_width': [8],  # Weight bit width. Default: 8.
    'weight_param_method': ['stats'],  # How scales/zero-point are determined. Default: stats.
    'weight_scale_precision': ['float_scale'
                              ],  # Whether scale is a float value or a po2. Default: po2.
    'weight_quant_type': ['sym'],  # Weight quantization type. Default: asym.
    'weight_quant_format': ['int'],  # Weight quantization type. Default: int.
    'weight_quant_granularity': [
        'per_group'],  # Granularity for scales/zero-point of weights. Default: per_group.
    'weight_group_dim': [
        None],  # Override default group_dim for groupsize quantization. Default: layer-dependant
    'weight_group_size': [128],  # Group size for per_group weight quantization. Default: 128.
    'quantize_weight_zero_point': [False],  # Quantize weight zero-point.
    'input_bit_width': [None],  # Input bit width. Default: None (disables input quantization).
    'input_quant_format': ['int'],  # Input quantization type. Default: int.
    'input_param_method': ['stats'],  # How scales/zero-point are determined. Default: stats.
    'input_scale_precision': ['float_scale'
                             ],  # Whether input scale is a float value or a po2. Default: float.
    'input_scale_type': ['static'],  # Whether input scale is a static value or a dynamic value.
    'input_quant_type': ['asym'],  # Input quantization type. Default: asym.
    'input_quant_granularity': [
        'per_tensor'],  # Granularity for scales/zero-point of inputs. Default: per_tensor.
    'input_group_size': [64],  # Group size for per_group input quantization. Default: 64.
    'quantize_input_zero_point': [False],  # Quantize input zero-point.
    'quantize_last_layer': [False],  # Quantize last nn.Linear layer.
    'gptq': [False],  # Apply GPTQ.
    'gpfq': [False],  # Apply GPFQ.
    'gpxq_act_order': [False],  # Apply GPXQ activation ordering.
    'gpxq_use_quant_activations': [False],  # Use quantized activations in GPXQ.
    'gpxq_create_weight_orig': [False],  # Create weight_orig in GPXQ.
    'act_calibration': [False],  # Apply activation calibration.
    'bias_corr': [False],  # Apply bias correction.
    'ln_affine_merge': [False],  # Merge LN affine params.
    'no_quantize': [False],  # Disable quantization.
    'no_float16': [False],  # Disable float16 as base datatype and switch to float32.
    'replace_mha': [False],  # Replace HuggingFace Attention with a quantizable version
    'weight_equalization': [
        False],  # Apply weight equalization. Relevant to ReLU based models (e.g. OPT).
    'act_equalization': [None],  # Apply activation equalization (SmoothQuant).
    'load_awq': [None],  # Load the awq search results.
    'export_target': [None],  # Model export.
    'export_prefix': [None],  # Path prefix to use for the various export flows.
    'checkpoint_name': [None],  # Filename to save checkpoint.
    'fuse_sequences': [False],  # Whether to merge the dataset sequences.
    'learned_round': [None, "auto_round"]  # Whether to use learned round. If `None`, RTN is used.
}

parser = argparse.ArgumentParser(description='PyTorch LLM PTQ Validation')
parser.add_argument('idx', type=int)
for option_name, option_value in OPTIONS_DEFAULT.items():
    if isinstance(option_value[0], bool):
        type_args = parse_bool
    else:
        type_args = partial(parse_type, default_type=type(option_value[0]))
    parser.add_argument(f'--{option_name}', default=option_value, nargs="+", type=type_args)


def main():
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    args.gpu = get_gpu_index(args.idx)
    print("Iter {}, GPU {}".format(args.idx, args.gpu))

    try:
        ptq_llm_models(args)
    except Exception as E:
        print("Exception at index {}: {}".format(args.idx, E))


def ptq_llm_models(args):
    # Generate all possible combinations, including invalid ones

    options = {k: getattr(args, k) for k, _ in OPTIONS_DEFAULT.items()}

    combinations = list(product(*options.values()))

    configs = []
    for combination in combinations:
        config_namespace = SimpleNamespace(
            **{k: v for k, v in zip(OPTIONS_DEFAULT.keys(), combination)})
        config_namespace = validate_config(config_namespace)
        if config_namespace.is_valid:
            configs.append(hashabledict(**config_namespace.__dict__))

    configs = unique(configs)

    if args.idx > len(configs) - 1:
        return

    config_namespace = SimpleNamespace(**configs[args.idx])
    print(config_namespace)

    if config_namespace.export_prefix is None:
        config_namespace.export_prefix = f"{config_namespace.model.replace('/', '--')}"

    if config_namespace.no_float16:
        dtype = torch.float32
    else:
        dtype = torch.float16

    kwargs = {"torch_dtype": dtype}

    if config_namespace.export_target == 'torch_qcdq':
        kwargs['torchscript'] = True

    print("Model loading...")
    model = AutoModelForCausalLM.from_pretrained(config_namespace.model, **kwargs)
    print("Model loaded.")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config_namespace.model)
    float_ppl = None
    quant_ppl = None

    if config_namespace.load_awq:
        from brevitas_examples.llm.llm_quant.awq.pre_quant import apply_awq
        awq_results = torch.load(config_namespace.load_awq, map_location="cpu")
        with CastFloat16ToFloat32():
            apply_awq(model, awq_results)

    require_fx = True if config_namespace.weight_equalization or config_namespace.act_equalization == 'fx' or config_namespace.ln_affine_merge else False

    # Load the data for calibration and evaluation.
    calibration_loader = get_dataset_for_model(
        config_namespace.model,
        dataset_name=config_namespace.dataset,
        tokenizer=tokenizer,
        nsamples=config_namespace.nsamples,
        seqlen=config_namespace.seqlen,
        split="train",
        seed=config_namespace.seed,
        require_fx=require_fx,
        device=None,
        fuse_sequences=config_namespace.fuse_sequences,
    )

    validation_loader = get_dataset_for_model(
        config_namespace.model,
        dataset_name=config_namespace.dataset,
        tokenizer=tokenizer,
        nsamples=config_namespace.nsamples,
        seqlen=config_namespace.seqlen,
        split="validation",
        seed=config_namespace.seed,
        require_fx=require_fx,
        device=None,
        fuse_sequences=config_namespace.fuse_sequences,
    )

    device = next(iter(model.parameters())).device
    print("Data loaded.")

    if config_namespace.eval:
        assert config_namespace.export_target != 'torch_qcdq', "TorchScript QCDQ export and Evaluation simultaneously"
        print("Float model eval...")
        model = offload_model(model)
        float_ppl = compute_perplexity(
            model,
            validation_loader,
            context_length=config_namespace.seqlen // 2,
            tokenizer=tokenizer)
        remove_hooks(model)
        print(f"Float perplexity ({config_namespace.dataset}): {float_ppl:.3f}")

    if require_fx:
        model = get_fx(model)

    # Apply LN affine merging before inserting MHA layers
    # since currently there is support only for merging into Linear
    if config_namespace.ln_affine_merge:
        print("Apply LN affine merge...")
        apply_layernorm_affine_merge(model, dtype)
        print("LN affine merge applied.")

    # Insert standard MHA layers when performing fx based weight/act equalization to avoid dealing
    # with all the variability in HF implementations
    if config_namespace.replace_mha:
        print("Replace HF MHA with quantizable variants...")
        model = replace_mha_with_quantizable_layers(model, dtype)
        print("Replacing done.")

    if config_namespace.weight_equalization:
        print("Apply weight equalization...")
        # In case of float16 model, we need to offload to account for missing ops
        model = offload_model(model)
        apply_weight_equalization(model)
        remove_hooks(model)
        print("Weight equalization applied.")

    if config_namespace.act_equalization is not None:
        offload_model(model)
        print("Apply act equalization (SmoothQuant)...")
        apply_act_equalization(model, config_namespace.act_equalization, calibration_loader)
        print("Act equalization applied.")
        remove_hooks(model)

    if not config_namespace.no_quantize:
        name_blacklist = []
        print("Applying model quantization...")
        linear_input_quant, weight_quant, input_quant, q_scaled_quant, k_transposed_quant, v_quant, attn_output_weights_quant = generate_quantizers(
            dtype=dtype,
            weight_bit_width=config_namespace.weight_bit_width,
            weight_param_method=config_namespace.weight_param_method,
            weight_scale_precision=config_namespace.weight_scale_precision,
            weight_quant_type=config_namespace.weight_quant_type,
            weight_quant_granularity=config_namespace.weight_quant_granularity,
            weight_group_size=config_namespace.weight_group_size,
            weight_group_dim=config_namespace.weight_group_dim,
            quantize_weight_zero_point=config_namespace.quantize_weight_zero_point,
            weight_quant_format=config_namespace.weight_quant_format,
            input_bit_width=config_namespace.input_bit_width,
            input_quant_format=config_namespace.input_quant_format,
            input_scale_precision=config_namespace.input_scale_precision,
            input_scale_type=config_namespace.input_scale_type,
            input_param_method=config_namespace.input_param_method,
            input_quant_type=config_namespace.input_quant_type,
            input_quant_granularity=config_namespace.input_quant_granularity,
            input_group_size=config_namespace.input_group_size,
            quantize_input_zero_point=config_namespace.quantize_input_zero_point,
            device=device)
        layer_map = generate_quant_maps(
            linear_input_quant=linear_input_quant,
            weight_quant=weight_quant,
            input_quant=input_quant,
            q_scaled_quant=q_scaled_quant,
            k_transposed_quant=k_transposed_quant,
            v_quant=v_quant,
            attn_output_weights_quant=attn_output_weights_quant,
            dtype=dtype,
            device=device,
            input_quant_format=config_namespace.input_quant_format,
            quantize_embedding=False)
        if not config_namespace.quantize_last_layer:
            name_blacklist += ["lm_head", "embed_out"]
        model = layerwise_quantize(
            model=model, compute_layer_map=layer_map, name_blacklist=name_blacklist)
        # Tie back first/last layer weights in case they got untied
        print("Model quantization applied.")

    # If any equalization has taken places, the embedding layer and the fully connected one are
    # not tied anymore, and they need to be treated as standalone, separate layers.
    # In all other cases we can tie them back so to preserve memory.
    if config_namespace.act_equalization is None and not require_fx:
        model.tie_weights()

    if config_namespace.bias_corr:
        model = add_zero_bias_to_linear(model)

    model = offload_model(model)

    if config_namespace.learned_round:
        print("Applying learned round...")
        apply_learned_round_learning_llm(model, calibration_loader)
        print("Learned round applied.")

    if config_namespace.act_calibration:
        print("Apply act calibration...")
        apply_calibration(model, calibration_loader)
        print("Act calibration applied.")

    if config_namespace.gptq:
        print("Applying GPTQ...")
        apply_gptq(
            model,
            calibration_loader,
            act_order=config_namespace.gpxq_act_order,
            use_quant_activations=config_namespace.gpxq_use_quant_activations,
            create_weight_orig=config_namespace.gpxq_create_weight_orig)
        print("GPTQ applied.")

    if config_namespace.gpfq:
        print("Applying GPFQ...")
        apply_gpfq(model, calibration_loader, act_order=config_namespace.gpxq_act_order)
        print("GPFQ applied.")

    if config_namespace.bias_corr:
        print("Applying bias correction...")
        apply_bias_correction(model, calibration_loader)
        print("Bias correction applied.")

    if config_namespace.eval:
        print("Model eval...")
        quant_ppl = compute_perplexity(
            model,
            validation_loader,
            context_length=config_namespace.seqlen // 2,
            tokenizer=tokenizer)
        print(f"Quantized perplexity ({config_namespace.dataset}): {quant_ppl:.3f}")
    remove_hooks(model)

    # Validate the quant_model on the validation dataloader
    print("Starting validation")

    column_names = [k.replace('_', ' ').capitalize() for k in config_namespace.__dict__.keys()] + [
        'FP perplexity', 'Quant perplexity', 'Torch version', 'Brevitas version']
    values = [v for _, v in config_namespace.__dict__.items()] + [
        float_ppl, quant_ppl, torch_version, brevitas_version]
    torchvision_df = pd.DataFrame([values], columns=column_names)

    folder = './multirun/' + str(args.idx)
    os.makedirs(folder, exist_ok=True)
    torchvision_df.to_csv(os.path.join(folder, 'RESULTS_LLM.csv'), index=False)


def validate_config(config_namespace):
    is_valid = True

    if not config_namespace.no_quantize:
        if config_namespace.gptq and config_namespace.gpfq:
            is_valid = False
        if config_namespace.export_target is not None:
            if config_namespace.input_quant_format != 'int':
                is_valid = False
        if config_namespace.export_target is not None and config_namespace.input_bit_width is not None:
            if config_namespace.input_scale_type != 'static':
                is_valid = False
        if config_namespace.export_target == 'sharded_torchmlir_group_weight':
            if config_namespace.weight_quant_granularity != 'per_group':
                is_valid = False
            if config_namespace.input_bit_width is not None:
                is_valid = False
            if config_namespace.quantize_weight_zero_point:
                is_valid = False
        if config_namespace.export_target == 'sharded_packed_torchmlir_group_weight':
            if config_namespace.weight_quant_granularity != 'per_group':
                is_valid = False
            if config_namespace.input_bit_width is not None:
                is_valid = False
            if config_namespace.quantize_weight_zero_point:
                is_valid = False
        if config_namespace.export_target == 'onnx_qcdq':
            if config_namespace.weight_quant_granularity == 'per_group':
                if config_namespace.input_bit_width is not None:
                    is_valid = False
            if config_namespace.weight_quant_type == 'asym':
                if not config_namespace.quantize_weight_zero_point:
                    is_valid = False
            if config_namespace.input_bit_width is not None and config_namespace.input_quant_type == 'asym':
                if not config_namespace.quantize_input_zero_point:
                    is_valid = False
        if config_namespace.export_target == 'torch_qcdq':
            if config_namespace.weight_quant_granularity == 'per_group':
                is_valid = False
            if config_namespace.weight_quant_type == 'asym':
                if not config_namespace.quantize_weight_zero_point:
                    is_valid = False
            if config_namespace.input_bit_width is not None and config_namespace.input_quant_type == 'asym':
                if not config_namespace.quantize_input_zero_point:
                    is_valid = False
        if config_namespace.input_bit_width and config_namespace.input_scale_type == 'static':
            if not config_namespace.act_calibration:
                is_valid = False
        if (config_namespace.weight_equalization or config_namespace.act_equalization == 'fx'):
            if config_namespace.replace_mha:
                if config_namespace.export_target == 'onnx_qcdq':
                    is_valid = False
            else:
                if config_namespace.export_target == 'torch_qcdq':
                    is_valid = False

    config_namespace.is_valid = is_valid
    return config_namespace


if __name__ == '__main__':
    main()
