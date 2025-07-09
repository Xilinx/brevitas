# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
import inspect
import os

import yaml

from brevitas_examples.imagenet_classification.ptq.ptq_imagenet_args import \
    create_args_parser as create_imagenet_ptq_args_parser
from brevitas_examples.llm.llm_args import create_args_parser as create_llm_args_parser
from brevitas_examples.stable_diffusion.stable_diffusion_args import \
    create_args_parser as create_sd_args_parser

ENTRYPOINT_ARGS = {
    "llm": create_llm_args_parser,
    "stable_diffusion": create_sd_args_parser,
    "imagenet_classification/ptq": create_imagenet_ptq_args_parser}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Specify the filename of the output config template')
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='If enabled, a template benchmark config is generated.')
    parser.add_argument(
        '--entrypoint',
        type=str,
        choices=list(ENTRYPOINT_ARGS.keys()),
        help='Specify the entrypoint for which to update the config template YAML')
    args = parser.parse_args()
    entrypoint_args = ENTRYPOINT_ARGS[args.entrypoint]()
    default_args = entrypoint_args.parse_args([])
    args_dict = default_args.__dict__
    del args_dict["config"]  # Config file cannot be specified via YAML
    # If --save-benchmark, a template benchmark config is generated instead
    if args.benchmark:
        args_dict = {k: [v] for k, v in args_dict.items()}
    # If --config is not specified, save as the default template within the selected entrypoint
    if args.config is None:
        config_path = ('benchmark', 'benchmark_template.yaml') if args.benchmark else (
            'config', 'default_template.yaml')
        args.config = os.path.join(
            os.path.dirname(inspect.getfile(ENTRYPOINT_ARGS[args.entrypoint])), *config_path)
    with open(args.config, 'w') as f:
        yaml.dump(args_dict, f)
