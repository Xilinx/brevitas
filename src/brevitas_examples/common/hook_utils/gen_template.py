# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser

import yaml

from brevitas_examples.llm.llm_args import create_llm_args_parser

ENTRYPOINT_ARGS = {"llm": create_llm_args_parser}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--config', type=str, help='Specify the filename of the output config template')
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
    with open(args.config, 'w') as f:
        yaml.dump(args_dict, f)
