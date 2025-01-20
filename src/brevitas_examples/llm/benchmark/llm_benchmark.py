from argparse import ArgumentParser
from argparse import Namespace
import itertools
import os
import re
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd
import randomname as rn
import yaml

from brevitas_examples.llm.main import instantiate_llm_parser
from brevitas_examples.llm.main import validate

# Set appropiately for your system
PYTHON_BIN = "CONDA_ENV_DIR/bin/python"
LLM_ENTRYPOINT = "BREVITAS_DIR/brevitas/src/brevitas_examples/llm/main.py"
RESULTS_FOLDER = "RESULTS_DIR"
CUDA_AVAILABLE_DEVICES = [0, 1]
NUM_GPUS_PER_PROCESS = 1


def run_args_bucket(id: int, args_dicts_bucket: List[Dict]):
    # Visible devices for the thread
    thread_cuda_visible_devices = ",".join(
        map(str, CUDA_AVAILABLE_DEVICES[id:id + NUM_GPUS_PER_PROCESS]))
    # Iterate over the combinations launching the LLM entrypoint
    for i in range(len(args_dicts_bucket)):
        print(f"Thread {id}, starting process {i+1}/{len(args_dicts_bucket)}")
        # Generate name for the experiment
        job_name = rn.get_name()
        job_folder = f"{RESULTS_FOLDER}/{job_name}"
        # Create folder to store the results of the experiment
        os.mkdir(job_folder)
        # Save yaml file for reproducibility
        with open(f"{job_folder}/config.yaml", 'w') as f:
            yaml.dump(args_dicts_bucket[0], f)
        # Run process
        stdout_file = open(f"{job_folder}/stdout.out", 'w')
        stderr_file = open(f"{job_folder}/stderr.out", 'w')
        process = subprocess.Popen(
            [PYTHON_BIN, LLM_ENTRYPOINT, "--config", f"{job_folder}/config.yaml"],
            env={"CUDA_VISIBLE_DEVICES": thread_cuda_visible_devices},
            stdout=stdout_file,
            stderr=stderr_file,
        )
        # Wait before starting a new process to prevent using the same GPUs
        process.wait()
        stdout_file.close()
        stderr_file.close()


def parse_config_args(args: List[str]) -> Namespace:
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=
        'Specify alternative default commandline args (e.g., config/default_template.yml). Default: %(default)s.'
    )
    return parser.parse_args(args)


def parse_results(columns: List[str], results_folder: str = RESULTS_FOLDER) -> pd.DataFrame:
    df = pd.DataFrame(columns=columns)
    for entry in os.scandir(results_folder):
        if entry.is_dir():
            # Get the identifier of the job
            job_name = os.path.basename(entry.path)
            # Retrieve the configuration from the YAML file
            with open(f"{results_folder}/{job_name}/config.yaml", 'r') as f:
                job_config = yaml.safe_load(f)
            # Load the log file
            with open(f"{results_folder}/{job_name}/stdout.out", 'r') as f:
                job_log = f.read()
                # Find the line containing Float PPL number
                float_ppl_line = re.search(r"Float perplexity \((.*?)\): (\d+\.\d+)", job_log)
                float_ppl = float(float_ppl_line.group(2)) if float_ppl_line is not None else None
                # Find the line containing Quant PPL number
                quant_ppl_line = re.search(r"Quantized perplexity \((.*?)\): (\d+\.\d+)", job_log)
                quant_ppl = float(quant_ppl_line.group(2)) if quant_ppl_line is not None else None
            # Add entry to DataFrame
            row_data = {
                "job_id": job_name, **job_config, "float_ppl": float_ppl, "quant_ppl": quant_ppl}
            df.loc[len(df)] = list(row_data.values())
    return df


if __name__ == "__main__":
    # Instantiate directory for storing the results
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
    if len(sys.argv) > 1:
        args = parse_config_args(sys.argv[1:])
        # Load argument combinations from specified YAML
        with open(args.config, 'r') as f:
            args_dict = yaml.safe_load(f)
    else:
        # Generate a YAML benchmark from default arguments
        llm_parser = instantiate_llm_parser()
        args_dict = {
            action.dest: [action.default] if action.choices is None else action.choices
            for action in llm_parser._actions}
        del args_dict["help"]  # Config file cannot be specified via YAML
        del args_dict["config"]  # Config file cannot be specified via YAML
        # Save YAML in the results folder
        with open(f"{RESULTS_FOLDER}/benchmark_config.yaml", 'w') as f:
            yaml.dump(args_dict, f)

    # Generate combinations of arguments
    args_keys, args_values = zip(*args_dict.items())
    # Retrieve argument combinations that are valid for the LLM entrypoint
    args_combinations = []
    for v in itertools.product(*args_values):
        args_combination = dict(zip(args_keys, v))
        try:
            # Check if the arguments are valid
            validate(SimpleNamespace(**args_combination))
            args_combinations.append(args_combination)
        except AssertionError:
            # Invalid configuration
            pass
    # Number of argument combinations
    num_combinations = len(args_combinations)
    num_buckets = len(CUDA_AVAILABLE_DEVICES) // NUM_GPUS_PER_PROCESS
    bucket_size = num_combinations // num_buckets
    # Split the combinations in differet buckets each belonging to a different thread
    args_combinations_buckets = [
        args_combinations[i * bucket_size:(i + 1) * bucket_size] for i in range(num_buckets)]

    # Instantiate threads to run the arguments in each bucket
    threads = []
    for i in range(num_buckets):
        thread = threading.Thread(
            target=run_args_bucket, args=(
                i,
                args_combinations_buckets[i],
            ))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Parse results
    df = parse_results(columns=["job_id"] + list(args_keys) + ["float_ppl", "quant_ppl"])
    df.to_csv(f"{RESULTS_FOLDER}/results.csv", index=False)
