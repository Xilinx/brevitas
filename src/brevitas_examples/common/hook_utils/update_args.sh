#!/bin/bash

for ENTRYPOINT_NAME in "llm" "stable_diffusion"; do
    BENCHMARK_CONFIG_PATH="src/brevitas_examples/${ENTRYPOINT_NAME}/benchmark/benchmark_config.yaml"
    ENTRYPOINT_ARGS_FILE="src/brevitas_examples/${ENTRYPOINT_NAME}/${ENTRYPOINT_NAME}_args.py"
    README_TEMPLATE_PATH="src/brevitas_examples/${ENTRYPOINT_NAME}/readme_template.md"
    README_PATH="src/brevitas_examples/${ENTRYPOINT_NAME}/README.md"
    TEMPLATE_CONFIG_PATH="src/brevitas_examples/${ENTRYPOINT_NAME}/config/default_template.yml"

    # Check if the arguments file was modified, as otherwise, the slowdown in the
    # pre-commit execution can be noticeable
    if [ "$1" = "--force" ] || git diff --name-only --staged | grep -q "$ENTRYPOINT_ARGS_FILE"; then
        python src/brevitas_examples/common/hook_utils/gen_readme.py --entrypoint "$ENTRYPOINT_NAME" --readme-template-path "$README_TEMPLATE_PATH" --readme-path "$README_PATH"
        python src/brevitas_examples/common/hook_utils/gen_template.py --entrypoint "$ENTRYPOINT_NAME" --config "$TEMPLATE_CONFIG_PATH"
        python src/brevitas_examples/common/hook_utils/cfg2benchmarkcfg.py --benchmark-config "$BENCHMARK_CONFIG_PATH" --config "$TEMPLATE_CONFIG_PATH"
    fi
done
