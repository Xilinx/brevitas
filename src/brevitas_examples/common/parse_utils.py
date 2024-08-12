"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import argparse
import re


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
