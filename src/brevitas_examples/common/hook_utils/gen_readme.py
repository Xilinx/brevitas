# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser

from jinja2 import Environment
from jinja2 import FileSystemLoader

from brevitas_examples.llm.llm_args import create_llm_args_parser

ENTRYPOINT_ARGS = {"llm": create_llm_args_parser}


def render_readme_template(
        argument_parser: ArgumentParser, readme_template_path: str, readme_path: str) -> None:
    # Set up the Jinja environment and load the template
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(readme_template_path)
    # Render the README.md template with the entrypoint arguments
    output = template.render(readme_help=argument_parser.format_help())
    # Save the rendered README.md
    with open(readme_path, 'w') as f:
        f.write(output + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        '--entrypoint',
        type=str,
        choices=list(ENTRYPOINT_ARGS.keys()),
        help='Specify the entrypoint for which to update the README')
    parser.add_argument(
        '--readme-template-path', type=str, help='Specify the path of the README template')
    parser.add_argument(
        '--readme-path', type=str, help='Specify the path in which to ouput the updated README')
    args = parser.parse_args()
    entrypoint_args = ENTRYPOINT_ARGS[args.entrypoint]()
    # Render the README template
    render_readme_template(
        argument_parser=entrypoint_args,
        readme_template_path=args.readme_template_path,
        readme_path=args.readme_path,
    )
