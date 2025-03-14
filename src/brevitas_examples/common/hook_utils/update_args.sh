#!/bin/bash
entrypoint_list=("llm,llm_args.py")

for tuple in "${entrypoint_list[@]}"; do
    # Split the tuple into individual elements
    IFS=',' read -r ENTRYPOINT_NAME ENTRYPOINT_ARGS_FILE <<< "$tuple"
    # Retrive paths for the specified entrypoint
    case $ENTRYPOINT_NAME in
        "llm")
            BENCHMARK_CONFIG_PATH="src/brevitas_examples/llm/benchmark/benchmark_config.yaml"
            README_TEMPLATE_PATH="src/brevitas_examples/llm/readme_template.md"
            README_PATH="src/brevitas_examples/llm/README.md"
            TEMPLATE_CONFIG_PATH="src/brevitas_examples/llm/config/default_template.yml"
            ;;
    esac
    # Check if the arguments file was modified, as otherwise, the slowdown in the
    # pre-commit execution can be noticeable
    if [[ "$1" == "--force" ]] || git diff --name-only --staged | grep -q "$ENTRYPOINT_ARGS_FILE" > /dev/null; then
        python src/brevitas_examples/common/hook_utils/gen_readme.py --entrypoint $ENTRYPOINT_NAME --readme-template-path $README_TEMPLATE_PATH --readme-path $README_PATH
        python src/brevitas_examples/common/hook_utils/gen_template.py --entrypoint $ENTRYPOINT_NAME --config $TEMPLATE_CONFIG_PATH
        python src/brevitas_examples/common/hook_utils/cfg2benchmarkcfg.py --benchmark-config $BENCHMARK_CONFIG_PATH --config $TEMPLATE_CONFIG_PATH
    fi
done
