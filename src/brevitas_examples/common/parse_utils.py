"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import argparse
from argparse import ArgumentParser
from argparse import Namespace
import re
from typing import Dict, List, Tuple

try:
    import yaml
except ImportError:
    yaml = None


class CustomValidator(object):

    def __init__(self, pattern):
        self._pattern = re.compile(pattern)

    def __call__(self, value):
        if not self._pattern.findall(value):
            raise argparse.ArgumentTypeError(
                "Argument has to match '{}'".format(self._pattern.pattern))
        return value


quant_format_validator = CustomValidator(r"int|e[1-8]m[1-8]")


def add_bool_arg(parser, name, default, help, str_true=False):
    dest = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    if str_true:
        group.add_argument('--' + name, dest=dest, type=str, help=help)
    else:
        group.add_argument('--' + name, dest=dest, action='store_true', help='Enable ' + help)
    group.add_argument('--no-' + name, dest=dest, action='store_false', help='Disable ' + help)
    parser.set_defaults(**{dest: default})


def override_defaults(args: List[str]) -> Dict:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=
        'Specify alternative default commandline args (e.g., config/default_template.yml). Default: %(default)s.'
    )
    known_args = parser.parse_known_args(args)[0]  # Returns a tuple
    if known_args.config is not None:
        assert yaml is not None, "The YAML config cannot be loaded, as the yaml package does not seem to be installed in your environment. Try running `pip install PyYAML`."
        with open(known_args.config, 'r') as f:
            defaults = yaml.safe_load(f)
    else:
        defaults = {}
    return defaults


def parse_args(parser: ArgumentParser,
               args: List[str],
               override_defaults: Dict = {}) -> Tuple[Namespace, List[str]]:
    if len(override_defaults) > 0:
        # Retrieve keys that are known to the parser
        parser_keys = set(map(lambda action: action.dest, parser._actions))
        # Extract the entries in override_defaults that correspond to keys not known to the parser
        extra_args_keys = [key for key in override_defaults.keys() if key not in parser_keys]
        # Remove all the keys in override_defaults that are unknown to the parser and, instead,
        # include them in args, as if they were passed as arguments to the command line.
        # This prevents the keys of HF TrainingArguments from being added as arguments to the parser.
        # Consequently, they will be part of the second value returned by parse_known_args (thus being
        # used as extra_args in quantize_llm)
        for key in extra_args_keys:
            args += [f"--{key}", str(override_defaults[key])]
            del override_defaults[key]
    parser.set_defaults(**override_defaults)
    return parser.parse_known_args(args)
