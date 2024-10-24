seq 0 10 | xargs -n1 -P3 -I{} sh -c 'HF_HUB_CACHE=/scratch/hf_models/ python llm_benchmark.py "$@"' _ {}
