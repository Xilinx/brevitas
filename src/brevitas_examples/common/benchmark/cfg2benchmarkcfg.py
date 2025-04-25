# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser

import yaml

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        required=True,
        help='Specify a configuration file (YAML) to use as a template for a benchmark config')
    parser.add_argument(
        '--benchmark-config',
        type=str,
        default=None,
        required=True,
        help='Specify the filename of the output benchmark config template')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        defaults = yaml.safe_load(f)
    benchmark_defaults = {k: [v] for k, v in defaults.items()}
    with open(args.benchmark_config, 'w') as f:
        yaml.dump(benchmark_defaults, f)
