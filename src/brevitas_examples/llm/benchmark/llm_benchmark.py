from argparse import ArgumentParser
from argparse import Namespace
import datetime
from functools import reduce
import itertools
import os
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Dict, List

import pandas as pd
import randomname as rn
import yaml

from brevitas_examples.llm.main import create_llm_args_parser
from brevitas_examples.llm.main import validate

# Set appropiately for your system
PYTHON_BIN = "CONDA_ENV_DIR/bin/python"
LLM_ENTRYPOINT = "BREVITAS_DIR/brevitas/src/brevitas_examples/llm/main.py"
RESULTS_FOLDER = "RESULTS_DIR"
CUDA_AVAILABLE_DEVICES = [0, 1]
NUM_GPUS_PER_PROCESS = 1
NUM_RETRIES = 1


def run_args_bucket(id: int, num_threads: int, args_dicts_queue: List[Dict]):
    # Visible devices for the thread
    thread_cuda_visible_devices = ",".join(
        map(str, CUDA_AVAILABLE_DEVICES[id:id + NUM_GPUS_PER_PROCESS]))
    # Provide ballpark estimates of remaining time
    mean_running_time = 0
    num_runs = 0
    # Iterate over the combinations launching the LLM entrypoint
    while True:
        try:
            # .pop is an atomic operation
            args_dict = args_dicts_queue.pop()
        except IndexError:
            break
        print(
            f"Thread {id}, remaining combinations {len(args_dicts_queue)}, remaining time: {'unknown' if num_runs == 0 else str(datetime.timedelta(seconds=int((len(args_dicts_queue) / num_threads + 1)*mean_running_time)))}"
        )
        # Generate name for the experiment
        job_name = rn.get_name()
        job_folder = f"{RESULTS_FOLDER}/{job_name}"
        # Create folder to store the results of the experiment
        os.mkdir(job_folder)
        # Save yaml file for reproducibility
        with open(f"{job_folder}/config.yaml", 'w') as f:
            yaml.dump(args_dict, f)
        # Enable reruning the process there was a crash
        num_retries = 0
        while num_retries < NUM_RETRIES:
            stdout_file = open(f"{job_folder}/stdout.out", 'w')
            stderr_file = open(f"{job_folder}/stderr.out", 'w')
            process = subprocess.Popen(
                [PYTHON_BIN, LLM_ENTRYPOINT, "--config", f"{job_folder}/config.yaml"],
                env={"CUDA_VISIBLE_DEVICES": thread_cuda_visible_devices},
                stdout=stdout_file,
                stderr=stderr_file,
            )
            # Wait before starting a new process to prevent using the same GPUs
            start_time = time.time()
            return_code = process.wait()
            end_time = time.time()
            running_time = end_time - start_time
            stdout_file.close()
            stderr_file.close()
            num_retries += 1
            # Dump information regarding the state of the run
            with open(f"{job_folder}/run_info.yaml", 'w') as f:
                yaml.dump({
                    "elapsed_time": running_time,
                    "return_code": return_code,
                    "retry_number": num_retries},
                          f)
            if return_code is not None and return_code == 0:
                # Update mean running time
                num_runs += 1
                mean_running_time = mean_running_time * (
                    num_runs - 1) / num_runs + running_time / num_runs
                break


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


def parse_results(results_folder: str = RESULTS_FOLDER) -> pd.DataFrame:
    row_data_list = []
    for entry in os.scandir(results_folder):
        if entry.is_dir():
            # Get the identifier of the job
            job_name = os.path.basename(entry.path)
            # Retrieve the configuration from the YAML file
            with open(f"{results_folder}/{job_name}/config.yaml", 'r') as f:
                job_config = yaml.safe_load(f)
            with open(f"{results_folder}/{job_name}/run_info.yaml", 'r') as f:
                job_info = yaml.safe_load(f)
            # Load the log file
            with open(f"{results_folder}/{job_name}/stdout.out", 'r') as f:
                job_log = f.read()
                # Find the line containing Float PPL number
                float_ppl_line = re.search(r"Float perplexity \((.*?)\): (\d+\.\d+)", job_log)
                float_ppl = float(float_ppl_line.group(2)) if float_ppl_line is not None else None
                # Find the line containing Quant PPL number
                quant_ppl_line = re.search(r"Quantized perplexity \((.*?)\): (\d+\.\d+)", job_log)
                quant_ppl = float(quant_ppl_line.group(2)) if quant_ppl_line is not None else None
                # Search for dictionary in log
                few_shot_eval_line = re.findall(r"({.*?})", job_log)
                # Retrieve last dictionary, in case other dictionaries were printed to the log
                few_shot_eval = eval(few_shot_eval_line[-1]) if len(few_shot_eval_line) > 0 else {}
            # Add entry to DataFrame
            row_data = {
                "job_id": job_name,
                **job_config,
                **job_info,
                "float_ppl": float_ppl,
                "quant_ppl": quant_ppl,
                **few_shot_eval}
            row_data_list.append(row_data)
    # Columns are obtained by computing the union of the sets of keys in row_data_list
    common_keys = ["job_id"] + list(job_config.keys()) + list(job_info.keys()) + [
        "float_ppl", "quant_ppl"]
    common_keys_set = set(common_keys)
    columns = common_keys + list(
        reduce(lambda x, y: x.union(y),
               [set(row_data.keys()) for row_data in row_data_list]).difference(common_keys_set))
    # Instantiate DataFrame to store the results
    df = pd.DataFrame(columns=columns)
    for row_data in row_data_list:
        # Fill missing columns with None
        df.loc[len(df)] = [row_data[key] if key in row_data else None for key in columns]
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
        llm_parser = create_llm_args_parser()
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
    num_threads = len(CUDA_AVAILABLE_DEVICES) // NUM_GPUS_PER_PROCESS
    # Instantiate threads to run the arguments in each bucket
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(
            target=run_args_bucket, args=(
                i,
                num_threads,
                args_combinations,
            ))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    # Parse results
    df = parse_results()
    df.to_csv(f"{RESULTS_FOLDER}/results.csv", index=False)
