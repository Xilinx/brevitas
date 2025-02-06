from argparse import ArgumentParser
from argparse import Namespace
import datetime
from functools import reduce
import itertools
import multiprocessing
from multiprocessing import Queue
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
RESULTS_FOLDER = "./src/brevitas_examples/llm/benchmark"
CUDA_AVAILABLE_DEVICES = [0, 1]
NUM_GPUS_PER_PROCESS = 1
NUM_RETRIES = 1


def _make_float(value):
    try:
        float_value = float(value)
        return float_value
    except ValueError:
        return value


def run_args_bucket_process(
        id: int, num_processes: int, cuda_visible_devices: str, args_dicts_queue: Queue):
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # Now import the LLM entrypoint, thus making sure that CUDA_VISIBLE_DEVICES
    # was set before importing torch
    from brevitas_examples.llm.main import quantize_llm

    # Provide ballpark estimates of remaining time
    mean_running_time = 0
    num_runs = 0
    # Keep references to original stdout and stderr
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    # Iterate over the combinations launching the LLM entrypoint
    while True:
        try:
            # Extract an element of the queue of combinations
            args_dict = args_dicts_queue.get(timeout=10.)
            if args_dict is None:
                break
        except Exception:
            break
        print(
            f"Process {id}, remaining combinations {args_dicts_queue.qsize()}, remaining time: {'unknown' if num_runs == 0 else str(datetime.timedelta(seconds=int((args_dicts_queue.qsize() / num_processes + 1)*mean_running_time)))}"
        )
        # TODO: Change so each process has an unique name, without relying
        # on the process id suffix
        job_name = f"{rn.get_name()}-{id}"
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
            # Redirect output to files
            sys.stdout = stdout_file
            sys.stderr = stderr_file
            # Wait before starting a new process to prevent using the same GPUs
            start_time = time.time()
            try:
                results, _ = quantize_llm(SimpleNamespace(**args_dict))
                results = {k: _make_float(v) for k, v in results.items()}
            except Exception as e:
                # Print exception to stderr, so it can be checked in log
                print(e, file=sys.stderr)
                results = None
            end_time = time.time()
            # Restore stdout and stderr
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            # Calculate elapsed time
            running_time = end_time - start_time
            stdout_file.close()
            stderr_file.close()
            num_retries += 1
            # Dump information regarding the state of the run
            with open(f"{job_folder}/run_results.yaml", 'w') as f:
                yaml.dump({
                    "elapsed_time": running_time,
                    "status": "crashed" if results is None else "succesful",
                    "retry_number": num_retries,
                    **(results if results is not None else {})},
                          f)
            if results is not None:
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
            with open(f"{results_folder}/{job_name}/run_results.yaml", 'r') as f:
                job_results = yaml.safe_load(f)
            # If the job was not succesful, try parsing the log
            if job_results["status"] == "crashed":
                # Load the log file
                with open(f"{results_folder}/{job_name}/stdout.out", 'r') as f:
                    job_log = f.read()
                    # Find the line containing Float PPL number
                    float_ppl_line = re.search(r"Float perplexity \((.*?)\): (\d+\.\d+)", job_log)
                    float_ppl = float(
                        float_ppl_line.group(2)) if float_ppl_line is not None else None
                    # Find the line containing Quant PPL number
                    quant_ppl_line = re.search(
                        r"Quantized perplexity \((.*?)\): (\d+\.\d+)", job_log)
                    quant_ppl = float(
                        quant_ppl_line.group(2)) if quant_ppl_line is not None else None
                    # Search for dictionary in log
                    few_shot_eval_line = re.findall(r"({.*?})", job_log)
                    # Retrieve last dictionary, in case other dictionaries were printed to the log
                    few_shot_eval = eval(
                        few_shot_eval_line[-1]) if len(few_shot_eval_line) > 0 else {}
                # Manually populate the results
                job_results = {
                    "elapsed_time": job_results["elapsed_time"],
                    "status": job_results["status"],
                    "retry_number": job_results["retry_number"],
                    "float_ppl": float_ppl,
                    "quant_ppl": quant_ppl,
                    **few_shot_eval,}
            # Add entry to DataFrame
            row_data = {"job_id": job_name, **job_config, **job_results}
            row_data_list.append(row_data)
    # Columns are obtained by computing the union of the sets of keys in row_data_list
    common_keys = ["job_id"] + list(job_config.keys()) + [
        "elapsed_time", "status", "retry_number", "float_ppl", "quant_ppl"]
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
    # A CUDA error message is issued when changing CUDA_VISIBLE_DEVICES
    # if processes are started in fork mode
    multiprocessing.set_start_method('spawn')
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
    q = Queue()
    args_combinations = []
    for v in itertools.product(*args_values):
        args_combination = dict(zip(args_keys, v))
        try:
            # Check if the arguments are valid
            validate(SimpleNamespace(**args_combination))
            args_combinations.append(args_combination)
            q.put(args_combination)
        except AssertionError:
            # Invalid configuration
            pass
    # Number of argument combinations
    num_processes = len(CUDA_AVAILABLE_DEVICES) // NUM_GPUS_PER_PROCESS
    # Instantiate threads to run the arguments in each bucket
    processes = []
    for i in range(num_processes):
        cuda_visible_devices = ",".join(
            map(str, CUDA_AVAILABLE_DEVICES[i:i + NUM_GPUS_PER_PROCESS]))
        process = multiprocessing.Process(
            target=run_args_bucket_process,
            args=(
                i,
                num_processes,
                cuda_visible_devices,
                q,
            ),
        )
        process.start()
        processes.append(process)

    # Wait for all threads to complete
    for process in processes:
        process.join()
    # Parse results
    df = parse_results()
    df.to_csv(f"{RESULTS_FOLDER}/results.csv", index=False)
