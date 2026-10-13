"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
"""

import argparse
from argparse import ArgumentParser
from argparse import Namespace
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

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


def parse_type(v, default_type):
    if v == 'None':
        return None
    else:
        return default_type(v)


def add_bool_arg(parser, name, default, help, str_true=False):
    dest = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    if str_true:
        group.add_argument('--' + name, dest=dest, type=str, help=help)
    else:
        group.add_argument('--' + name, dest=dest, action='store_true', help='Enable ' + help)
    group.add_argument('--no-' + name, dest=dest, action='store_false', help='Disable ' + help)
    parser.set_defaults(**{dest: default})


def create_entrypoint_args_parser(description: Optional[str] = None) -> ArgumentParser:
    parser = ArgumentParser(description=description)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=
        'Specify alternative default commandline args (e.g., config/default_template.yml). Default: %(default)s.'
    )
    return parser


def override_defaults(args: List[str]) -> Dict[str, Any]:
    parser = ArgumentParser(add_help=False)
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


class AlgorithmArgumentParser:
    """Base class for per-algorithm argument groups.

    Each flag is prefixed so algorithms cannot collide: ``suffix`` becomes the
    CLI option ``--{prefix}-{suffix}`` and the attribute ``{prefix}_{suffix}``.
    Subclasses define arguments in :meth:`add_arguments` and cross-argument
    checks in :meth:`validate`, using the ``_flag``/``_dest``/``_add``/
    ``_add_bool`` helpers so the prefix is never hardcoded.
    """

    #: Default prefix; subclasses should override.
    prefix: str = ""

    # -- naming helpers -------------------------------------------------------

    @staticmethod
    def _flag(prefix: str, suffix: str) -> str:
        """CLI flag name (without ``--``): ``{prefix}-{suffix}``."""
        suffix = suffix.strip('-')
        return f"{prefix}-{suffix}"

    @staticmethod
    def _dest(prefix: str, suffix: str) -> str:
        """Namespace attribute name: ``{prefix}_{suffix}``."""
        return AlgorithmArgumentParser._flag(prefix, suffix).replace('-', '_')

    @staticmethod
    def _add_bool(
            parser: ArgumentParser, prefix: str, suffix: str, default: bool, help: str) -> None:
        # add_bool_arg derives dest from the flag name, so the prefixed flag
        # yields the prefixed dest for free.
        add_bool_arg(
            parser, AlgorithmArgumentParser._flag(prefix, suffix), default=default, help=help)

    @staticmethod
    def _add(parser: ArgumentParser, prefix: str, suffix: str, **kwargs) -> None:
        parser.add_argument(
            '--' + AlgorithmArgumentParser._flag(prefix, suffix),
            dest=AlgorithmArgumentParser._dest(prefix, suffix),
            **kwargs)

    # -- interface ------------------------------------------------------------

    @staticmethod
    def add_arguments(parser: ArgumentParser, prefix: str) -> None:
        raise NotImplementedError

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None, prefix: str = "") -> None:
        raise NotImplementedError
