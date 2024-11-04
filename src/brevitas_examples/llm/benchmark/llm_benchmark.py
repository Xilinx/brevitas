"""
Copyright (C) 2024,     Advanced Micro Devices, Inc.
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of AMD, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import argparse
from functools import partial
from itertools import product
import os
from types import SimpleNamespace

import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from brevitas import __version__ as brevitas_version
from brevitas import config
from brevitas import torch_version
from brevitas_examples.imagenet_classification.ptq.utils import get_gpu_index
# LLM example depends on optimum-amd, which requires PyTorch>=2.2
from brevitas_examples.llm.main import main as main_llm
from brevitas_examples.llm.main import validate

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
    'dataset': ['wikitext2'],  # Dataset to use for quantization (default: wikitext2)
    'gpxq_block_name': [None],  # Block name for faster GPxQ optimization. Default: None
    'weight_bit_width': [8],  # Weight bit width. Default: 8.
    'weight_param_method': ['stats'],  # How scales/zero-point are determined. Default: stats.
    'weight_scale_precision': ['float_scale'
                              ],  # Whether scale is a float value or a po2. Default: po2.
    'weight_quant_type': ['sym'],  # Weight quantization type. Default: asym.
    'weight_quant_format': ['int'],  # Weight quantization type. Default: int.
    'weight_quant_granularity': [
        'per_group'],  # Granularity for scales/zero-point of weights. Default: per_group.
    'scale_rounding_func_type': [None],  # Rounding function to use with Po2 scale. Default: None.
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
    'gpxq_act_order': [False],  # Apply GPxQ activation ordering.
    'gpxq_use_quant_activations': [False],  # Use quantized activations in GPxQ.
    'gpxq_create_weight_orig': [False],  # Create weight_orig in GPxQ.
    'gpxq_max_accumulator_bit_width': [None],  # Maximum accumulator bit width for GPxQ using AXE.
    'gpxq_max_accumulator_tile_size': [None],  # Maximum accumulator tile size for GPxQ using AXE.
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
    'learned_round': [None, "auto_round"],  # Whether to use learned round. If `None`, RTN is used.
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

    # Generate all possible configurations, including invalid ones
    options = {k: getattr(args, k) for k, _ in OPTIONS_DEFAULT.items()}
    combinations = list(product(*options.values()))
    configs = []
    for combination in combinations:
        config_namespace = SimpleNamespace(
            **{k: v for k, v in zip(OPTIONS_DEFAULT.keys(), combination)})
        try:
            validate(config_namespace)
            configs.append(hashabledict(**config_namespace.__dict__))
        except AssertionError:
            # Invalid configuration
            pass

    configs = unique(configs)

    if args.idx > len(configs) - 1:
        return

    config_namespace = SimpleNamespace(**configs[args.idx])
    args.gpu = get_gpu_index(args.idx)
    print("Iter {}, GPU {}".format(args.idx, args.gpu))

    try:
        float_ppl, quant_ppl, _ = main_llm(config_namespace)

        # Results are saved in CSV
        column_names = [k.replace('_', ' ').capitalize() for k in config_namespace.__dict__.keys()
                       ] + [
                           'FP perplexity', 'Quant perplexity', 'Torch version', 'Brevitas version']
        values = [v for _, v in config_namespace.__dict__.items()] + [
            float_ppl, quant_ppl, torch_version, brevitas_version]
        llm_df = pd.DataFrame([values], columns=column_names)

        folder = './multirun/' + str(args.idx)
        os.makedirs(folder, exist_ok=True)
        llm_df.to_csv(os.path.join(folder, 'RESULTS_LLM.csv'), index=False)

    except Exception as E:
        print("Exception at index {}: {}".format(args.idx, E))


if __name__ == '__main__':
    main()
