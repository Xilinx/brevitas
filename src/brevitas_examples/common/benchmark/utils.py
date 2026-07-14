# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC
from abc import abstractmethod
from argparse import Action
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
import datetime
from functools import reduce
import hashlib
import itertools
import multiprocessing
from multiprocessing import Queue
import os
import pprint
import random
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
import yaml

# A single benchmark experiment: (parsed known args, extra CLI args, full arg dict)
Experiment = Tuple[SimpleNamespace, List[str], Dict[str, Any]]


class EntryPointUtils(ABC):
    """Encapsulates everything specific to a benchmark entry point.

    An entry point knows how to run a single experiment (`entrypoint_main`),
    validate an argument combination (`validate`), parse metrics out of a job log
    (`parse_log`), and it exposes the argument parser for the underlying script
    (`argument_parser`) as well as the metrics it produces (`eval_metrics`).
    """

    @staticmethod
    @abstractmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def validate(args: Namespace, extra_args: List[str]) -> None:
        pass

    @staticmethod
    @abstractmethod
    def entrypoint_main(args: Namespace,
                        extra_args: List[str],
                        job_folder: Optional[str] = None) -> Tuple[Dict, Any]:
        pass

    @property
    @abstractmethod
    def argument_parser() -> ArgumentParser:
        pass

    @property
    @abstractmethod
    def eval_metrics() -> List[str]:
        pass


class SearchUtils(ABC):
    """Encapsulates a search strategy over the entry point's argument space.

    A search strategy does not know anything about a specific entry point. Any
    data it needs from the entry point (its argument parser and its validation
    function) is passed in explicitly by the :class:`BenchmarkUtils` coordinator.

    Each search strategy defines how an argument's default value is represented
    in its config format via :meth:`_default_action_handler`; the actual reading
    of the config file is owned by the :class:`BenchmarkUtils` coordinator.
    """

    @classmethod
    @abstractmethod
    def _default_action_handler(cls, action: Action) -> Any:
        pass

    @classmethod
    def script_parser(cls) -> ArgumentParser:
        parser = ArgumentParser()
        parser.add_argument(
            '--config',
            type=str,
            required=True,
            help='Specify YAML with argument combinations (e.g., benchmark/benchmark_config.yml)')
        parser.add_argument(
            '--results-folder',
            type=str,
            default="./",
            help='Folder to store the experiment results. Default: %(default)s.')
        parser.add_argument(
            '--gpus',
            type=str,
            default="0",
            help=
            'Specify the identifiers of the GPUs to use in a comma-separated list. Default: %(default)s.'
        )
        parser.add_argument(
            '--num-gpus-per-process',
            type=int,
            default=1,
            help=
            'Number of GPUs to each for running each argument combination. Default: %(default)s.')
        parser.add_argument(
            '--max-num-retries',
            type=int,
            default=1,
            help=
            'Number of retries for each argument combination in case a crash happens. Default: %(default)s.'
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Whether to skip running experiments (default: %(default)s).",
        )
        return parser

    @classmethod
    @abstractmethod
    def gen_search_space(
            cls,
            args_dict: Dict[str, Any],
            script_args: Namespace,
            entrypoint_parser: ArgumentParser,
            validate_fn: Callable[[Namespace, List[str]], None]) -> List[Experiment]:
        pass

    @classmethod
    def print_benchmark_summary(
            cls,
            args_dict: Dict[str, Any],
            args_queue: List[Experiment],
            script_args: Namespace,
            entrypoint_parser: ArgumentParser) -> None:
        print(f"Num. experiments: {len(args_queue)}")
        _print_config_block("Benchmark args.:", vars(script_args))
        if len(args_queue) == 0:
            return
        # Per-key summary of the search space (representation is subclass-specific).
        args_combinations = cls._collect_arg_combinations(args_dict, args_queue)
        # Drop args held constant at the entry point's default value.
        defaults = {action.dest: action.default for action in entrypoint_parser._actions}
        for key in list(args_combinations):
            if key not in defaults:
                continue
            is_constant, value = cls._constant_value(args_combinations[key])
            if is_constant and _as_hashable(value) == _as_hashable(defaults[key]):
                del args_combinations[key]
        non_default = {
            f"--{key.replace('_', '-')}": cls._format_combination(value) for key,
            value in args_combinations.items()}
        _print_config_block("Non-default args.:", non_default)

    # Per-key representation of the search space, keyed by argument name.
    @classmethod
    @abstractmethod
    def _collect_arg_combinations(cls, args_dict: Dict[str, Any],
                                  args_queue: List[Experiment]) -> Dict[str, Any]:
        pass

    # Whether a key is held constant and, if so, the constant value (for default
    # comparison). Returns (is_constant, value).
    @classmethod
    @abstractmethod
    def _constant_value(cls, combination: Any) -> Tuple[bool, Any]:
        pass

    # Display representation of a key's combination for the summary.
    @classmethod
    @abstractmethod
    def _format_combination(cls, combination: Any) -> Any:
        pass


class GridSearchUtils(SearchUtils):

    @classmethod
    def _default_action_handler(cls, action: Action) -> List[Any]:
        # Standardizes an argument for the given search class when it is not provided
        return [action.default]

    @classmethod
    def script_parser(cls) -> ArgumentParser:
        parser = super().script_parser()
        parser.add_argument(
            '--start-index',
            type=int,
            default=0,
            help=
            'Index from which to start current run. Note, the index is inclusive, e.g., a value of 3 will allow all processes from 3 onwards to run (default: %(default)s).'
        )
        parser.add_argument(
            '--end-index',
            type=int,
            default=-1,
            help=
            'Index from which to end current run. Note, the index is exclusive, e.g., a value of 10 will allow all processes from 0-9 to run.0 A negative value runs all jobs from `--start-index` (default: %(default)s).'
        )
        parser.add_argument(
            '--shuffle-seed',
            type=int,
            default=None,
            help=
            'The seed to use to shuffle the jobs. If None, no shuffling will be applied. Default: %(default)s.'
        )
        return parser

    @classmethod
    def gen_search_space(
            cls,
            args_dict: Dict[str, Any],
            script_args: Namespace,
            entrypoint_parser: ArgumentParser,
            validate_fn: Callable[[Namespace, List[str]], None]) -> List[Experiment]:
        # Generate combinations of arguments
        args_keys, args_values = zip(*args_dict.items())
        # Extract the keys that are known to the argument parser
        parser_keys = set(action.dest for action in entrypoint_parser._actions)
        # Retrieve argument combinations that are valid for the entrypoint
        exp_queue = []
        for v in itertools.product(*args_values):
            args_dict = dict(zip(args_keys, v))
            try:
                # Separate the arguments that are known to the parser and the extra
                # arguments that are used, for instance, in rotation optimization
                args = {}
                extra_args = []
                for key, value in args_dict.items():
                    if key in parser_keys:
                        args[key] = value
                    else:
                        extra_args += [f"--{key.replace('_', '-')}", str(value)]
                args = SimpleNamespace(**args)
                # Only keep valid configurations
                validate_fn(args, extra_args)
                exp_queue.append((args, extra_args, args_dict))
            except AssertionError:
                # Invalid configuration
                pass
        if script_args.shuffle_seed is not None:
            random.seed(script_args.shuffle_seed)
            random.shuffle(exp_queue)
        start_index = script_args.start_index
        end_index = script_args.end_index if script_args.end_index > 0 else len(exp_queue)
        exp_queue = exp_queue[start_index:end_index]
        return exp_queue

    @classmethod
    def _collect_arg_combinations(cls, args_dict: Dict[str, Any],
                                  args_queue: List[Experiment]) -> Dict[str, Any]:
        # Collect the set of observed values per argument across all experiments.
        args_combinations = defaultdict(set)
        for _, _, combination_dict in args_queue:
            for key, value in combination_dict.items():
                args_combinations[key].add(_as_hashable(value))
        return args_combinations

    @classmethod
    def _constant_value(cls, combination: Any) -> Tuple[bool, Any]:
        # An argument is constant if only a single value was observed.
        if len(combination) == 1:
            return True, next(iter(combination))
        return False, None

    @classmethod
    def _format_combination(cls, combination: Any) -> Any:
        return maybe_sort_values(combination)


# A node that samples a value for a single benchmark argument. Concrete node
# types are keyed by `rand_type` (as in the YAML), auto-registered via
# `__init_subclass__`, and built from parsed YAML by `from_config`.
class RandomArgNode(ABC):
    # Maps `rand_type` -> concrete node class.
    _registry: Dict[str, Type["RandomArgNode"]] = {}

    # Abstract attribute: concrete types override this with a plain string
    # (e.g. `rand_type = "const"`); otherwise the class stays abstract.
    @property
    @abstractmethod
    def rand_type(self) -> str:
        ...

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        rand_type = cls.rand_type  # accessed on the class
        # Not overridden -> still abstract; skip registration and let ABC block
        # instantiation.
        if getattr(rand_type, "__isabstractmethod__", False):
            return
        # Must be a string...
        if not isinstance(rand_type, str):
            raise TypeError(
                f"{cls.__name__}.rand_type must be a str, got {type(rand_type).__name__}")
        # ...and unique (pop the key from `_registry` to replace a built-in).
        if rand_type in RandomArgNode._registry:
            existing = RandomArgNode._registry[rand_type]
            raise TypeError(
                f"rand_type {rand_type!r} for {cls.__name__} is already registered "
                f"by {existing.__name__}")
        RandomArgNode._registry[rand_type] = cls

    def __init__(self, rand_values: Any) -> None:
        self.rand_values = rand_values

    @abstractmethod
    def value(self) -> Any:
        pass

    @classmethod
    def from_config(cls, rand_type: str, rand_values: Any) -> "RandomArgNode":
        # Build the node registered for `rand_type` (fail fast on unknown types).
        if rand_type not in cls._registry:
            raise ValueError(
                f"{rand_type} is not a valid random type. "
                f"Choices are: {sorted(cls._registry)}")
        return cls._registry[rand_type](rand_values)


class ConstNode(RandomArgNode):
    rand_type = "const"

    def value(self) -> Any:
        return self.rand_values

    def __str__(self) -> str:
        return f"type: {self.rand_type}, value: {self.rand_values}"


class ChoicesNode(RandomArgNode):
    rand_type = "choices"

    def value(self) -> Any:
        return random.choice(self.rand_values)

    def __str__(self) -> str:
        return f"type: {self.rand_type}, values: {self.rand_values}"


# Shared base for range ([min, max]) node types; validates bounds on construction.
class _RangeNode(RandomArgNode):

    def __init__(self, rand_values: Any) -> None:
        super().__init__(rand_values)
        # Range types require exactly [min, max] with numeric, ordered bounds.
        try:
            low, high = rand_values
        except (TypeError, ValueError):
            raise ValueError(
                f"'{self.rand_type}' requires rand_values to be a [min, max] pair, "
                f"got: {rand_values!r}")
        if not (isinstance(low, (int, float)) and isinstance(high, (int, float))):
            raise ValueError(f"'{self.rand_type}' bounds must be numeric, got: {rand_values!r}")
        if low > high:
            raise ValueError(f"'{self.rand_type}' requires min <= max, got: min={low}, max={high}")

    def __str__(self) -> str:
        return f"type: {self.rand_type}, min: {self.rand_values[0]}, max: {self.rand_values[1]}"


class LinearNode(_RangeNode):
    rand_type = "linear"

    def value(self) -> float:
        return random.uniform(self.rand_values[0], self.rand_values[1])


class Log2Node(_RangeNode):
    rand_type = "log2"

    def __init__(self, rand_values: Any) -> None:
        super().__init__(rand_values)
        # log2 sampling takes np.log2 of both bounds, which requires them > 0.
        low, high = rand_values
        if low <= 0 or high <= 0:
            raise ValueError(
                f"'{self.rand_type}' requires strictly positive bounds, got: {rand_values!r}")

    def value(self) -> float:
        return float(
            2 ** random.uniform(np.log2(self.rand_values[0]), np.log2(self.rand_values[1])))


class Exp2Node(_RangeNode):
    rand_type = "exp2"

    def value(self) -> float:
        return float(np.log2(random.uniform(2 ** self.rand_values[0], 2 ** self.rand_values[1])))


class RandomSearchUtils(SearchUtils):

    @classmethod
    def _default_action_handler(cls, action: Action) -> Dict[str, Any]:
        # Standardizes an argument for the given search class when it is not provided.
        return {"rand_type": "const", "rand_values": action.default}

    @classmethod
    def script_parser(cls) -> ArgumentParser:
        parser = super().script_parser()
        parser.add_argument(
            '--num-experiments',
            type=int,
            default=1,
            help='Number of random experiments to generate (default: %(default)s).')
        parser.add_argument(
            '--max-experimental-configs',
            type=int,
            default=100000,
            help=
            'Maximum number of experimental configurations to try before giving up (default: %(default)s).'
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=0,
            help='The seed to use for the search (default: %(default)s).')
        return parser

    @classmethod
    def gen_search_space(
            cls,
            args_dict: Dict[str, Any],
            script_args: Namespace,
            entrypoint_parser: ArgumentParser,
            validate_fn: Callable[[Namespace, List[str]], None]) -> List[Experiment]:
        generator_dict = {k: RandomArgNode.from_config(**v) for k, v in args_dict.items()}
        # Extract the keys that are known to the argument parser
        parser_keys = set(action.dest for action in entrypoint_parser._actions)
        # Retrieve argument combinations that are valid for the entrypoint
        exp_queue = []
        for i in range(script_args.max_experimental_configs):
            if len(exp_queue) >= script_args.num_experiments:
                break
            args_dict = {k: v.value() for k, v in generator_dict.items()}
            try:
                # Separate the arguments that are known to the parser and the extra
                # arguments that are used, for instance, in rotation optimization
                args = {}
                extra_args = []
                for key, value in args_dict.items():
                    if key in parser_keys:
                        args[key] = value
                    else:
                        extra_args += [f"--{key.replace('_', '-')}", str(value)]
                args = SimpleNamespace(**args)
                # Only keep valid configurations
                validate_fn(args, extra_args)
                exp_queue.append((args, extra_args, args_dict))
            except AssertionError:
                # Invalid configuration
                pass
        return exp_queue

    @classmethod
    def _collect_arg_combinations(cls, args_dict: Dict[str, Any],
                                  args_queue: List[Experiment]) -> Dict[str, Any]:
        # Reconstruct the per-argument random nodes from the standardized config.
        return {k: RandomArgNode.from_config(**v) for k, v in args_dict.items()}

    @classmethod
    def _constant_value(cls, combination: Any) -> Tuple[bool, Any]:
        # A random node is constant only if its rand_type is "const".
        if combination.rand_type == "const":
            return True, combination.value()
        return False, None

    @classmethod
    def _format_combination(cls, combination: Any) -> Any:
        return str(combination)


def _make_float(value: Any) -> Any:
    try:
        float_value = float(value)
        return float_value
    except Exception:
        return value


def _print_config_block(title: str, config: Dict) -> None:
    """Print a titled config block. Central place for config printing preferences."""
    print(title)
    pprint.pp(config)


# Ensures that the bytestring is the same irrespective
# of the order in which the keys are added to the dictionary
def _dict_to_bytes(dictionary: Dict) -> bytes:
    sorted_dict = {}
    for key in sorted(dictionary):
        sorted_dict[key] = dictionary[key]
    return str(sorted_dict).encode('utf-8')


# Not used at the moment, but kept for reference
def args_dict_to_command(entrypoint_parser: ArgumentParser, args_dict: Dict) -> str:
    from argparse import _StoreAction
    from argparse import _StoreTrueAction

    # Save actions from the argument parser
    args_parser_dict = {action.dest: action for action in entrypoint_parser._actions}
    # Iterate over the combinations
    command_options = []
    for key, value in args_dict.items():
        if key in args_parser_dict:
            action = args_parser_dict[key]
            if isinstance(action, _StoreAction):
                if value != action.default:
                    command_options += [f"--{key.replace('_', '-')}", str(value)]
            elif isinstance(action, _StoreTrueAction):
                if value:
                    command_options += [f"--{key.replace('_', '-')}"]
        else:
            command_options += [f"--{key.replace('_', '-')}", str(value)]
    return " ".join(command_options)


def run_args_bucket_process(
        main_entrypoint: Callable,
        id: int,
        num_processes: int,
        cuda_visible_devices: str,
        results_folder: str,
        max_num_retries: int,
        args_queue: Queue):
    # Set visible devices
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["HIP_VISIBLE_DEVICES"] = cuda_visible_devices
    # Imports are deferred to ensure that CUDA is not initialized
    # in the main process
    from brevitas import __version__ as brevitas_version
    from brevitas import torch_version

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
            args_tuple = args_queue.get(timeout=10.)
            if args_tuple is not None:
                args, extra_args, args_dict = args_tuple
            else:
                break
        except Exception:
            break
        job_name = f"{hashlib.md5(_dict_to_bytes(args_dict)).hexdigest()}"
        job_folder = f"{results_folder}/{job_name}"
        remaining_time = 'unknown'
        if num_runs != 0:
            remaining_time = str(
                datetime.timedelta(
                    seconds=int((args_queue.qsize() / num_processes + 1) * mean_running_time)))
        print(
            f"Job: {job_name}, process: {id}, gpu(s): {cuda_visible_devices}, remaining combinations: {args_queue.qsize()}, remaining time: {remaining_time}"
        )
        # Check if a folder for the experiment already exists. In case the
        # experiment was successful before, do not try to run again
        if os.path.isdir(job_folder):
            try:
                with open(f"{job_folder}/run_results.yaml", 'r') as f:
                    job_results = yaml.safe_load(f)
                if job_results["status"] == "successful":
                    # Skip experiment
                    continue
            except Exception:
                pass
        else:
            os.mkdir(job_folder)
        # Save yaml file for reproducibility
        with open(f"{job_folder}/config.yaml", 'w') as f:
            yaml.dump(args_dict, f)
        # Enable reruning the process there was a crash
        num_retries = 0
        while num_retries < max_num_retries:
            stdout_file = open(f"{job_folder}/stdout.out", 'w')
            stderr_file = open(f"{job_folder}/stderr.out", 'w')
            # Redirect output to files
            sys.stdout = stdout_file
            sys.stderr = stderr_file
            # Record the wall-clock elapsed time when running the LLM entrypoint
            start_time = time.time()
            try:
                results, _ = main_entrypoint(args, extra_args, job_folder)
                results = {k: _make_float(v) for k, v in results.items()}
            except Exception:
                # Print exception to stderr, so it can be checked in log
                print(traceback.format_exc(), file=sys.stderr)
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
            # Dump information with the state and results of the run
            with open(f"{job_folder}/run_results.yaml", 'w') as f:
                yaml.dump({
                    "elapsed_time": running_time,
                    "status": "crashed" if results is None else "successful",
                    "retry_number": num_retries,
                    "brevitas_version": brevitas_version,
                    "torch_version": str(torch_version),
                    **(results if results is not None else {})},
                          f)
            if results is not None:
                # Update mean running time and move to next combination
                num_runs += 1
                mean_running_time = mean_running_time * (
                    num_runs - 1) / num_runs + running_time / num_runs
                break


def maybe_sort_values(values):
    try:
        sorted_values = list(sorted(values))
    except Exception:
        # Fails if the list contains None
        sorted_values = list(values)
    return sorted_values


def _as_hashable(value: Any) -> Any:
    # Lists are unhashable and can't be compared to tuple defaults; normalize both.
    return tuple(value) if isinstance(value, list) else value


class BenchmarkUtils:
    """Coordinator composing an :class:`EntryPointUtils` and a :class:`SearchUtils`.

    Subclasses provide the two members as class attributes::

        class LLMGridBenchmark(BenchmarkUtils):
            entry_point_utils = LLMEntryPointUtils
            search_utils = GridSearchUtils

    ``BenchmarkUtils`` owns the interaction between the two members: it extracts
    the data the search needs from the entry point (its argument parser and its
    validation function) and feeds the resulting experiment queue back into the
    entry point for execution.
    """

    entry_point_utils: Type[EntryPointUtils] = None
    search_utils: Type[SearchUtils] = None

    @classmethod
    def standardize_args(cls, script_args: Namespace) -> Dict[str, Any]:
        # Read the benchmark config file and construct a full set of arguments,
        # converting between the entry point's argument space and the search's
        # config format. The generic reading of the config is owned here, while
        # the per-argument default representation is delegated to the search.
        if script_args.config is None:
            raise ValueError("Config file not specified")
        with open(script_args.config, 'r') as f:
            args_dict = yaml.safe_load(f)
        # Add defaults if only a subset of keys are specified
        for action in cls.entry_point_utils.argument_parser._actions:
            if action.dest not in args_dict:
                args_dict[action.dest] = cls.search_utils._default_action_handler(action)
        return args_dict

    @classmethod
    def parse_results(cls, results_folder: str) -> pd.DataFrame:
        entry_point_utils = cls.entry_point_utils
        row_data_list = []
        job_config = None
        for entry in os.scandir(results_folder):
            if entry.is_dir() and entry.name not in ["__pycache__"]:
                # Get the identifier of the job
                job_name = os.path.basename(entry.path)
                # Retrieve the configuration from the YAML file
                with open(f"{results_folder}/{job_name}/config.yaml", 'r') as f:
                    job_config = yaml.safe_load(f)
                try:
                    with open(f"{results_folder}/{job_name}/run_results.yaml", 'r') as f:
                        job_results = yaml.safe_load(f)
                except Exception:
                    # Failsafe if entrypoint failed in a way that brings down the whole process
                    job_results = {
                        "status": "crashed",
                        "elapsed_time": -1.,
                        "retry_number": -1.,
                        "brevitas_version": -1.,
                        "torch_version": -1.,}
                # If the job was not succesful, try parsing the log
                if job_results["status"] == "crashed":
                    # Load the log file
                    with open(f"{results_folder}/{job_name}/stdout.out", 'r') as f:
                        job_log = f.read()
                        # Parse results from log
                        job_log_results = entry_point_utils.parse_log(job_log)
                    # Manually populate the results
                    job_results = {
                        "elapsed_time": job_results["elapsed_time"],
                        "status": job_results["status"],
                        "retry_number": job_results["retry_number"],
                        "brevitas_version": job_results["brevitas_version"],
                        "torch_version": job_results["torch_version"],
                        **job_log_results,}
                # Add entry to DataFrame
                row_data = {"job_id": job_name, **job_config, **job_results}
                row_data_list.append(row_data)
        if job_config is not None:
            # Columns are obtained by computing the union of the sets of keys in row_data_list,
            # since, for instance, some jobs might have crashed before completing the LM eval
            common_keys = ["job_id"] + list(job_config.keys()) + [
                "elapsed_time", "status", "retry_number", "brevitas_version", "torch_version"
            ] + entry_point_utils.eval_metrics
            common_keys_set = set(common_keys)
            columns = common_keys + list(
                reduce(
                    lambda x, y: x.union(y), [set(row_data.keys()) for row_data in row_data_list
                                             ]).difference(common_keys_set))
            # Instantiate DataFrame to store the results
            df = pd.DataFrame(columns=columns)
            for row_data in row_data_list:
                # Fill missing columns with None
                df.loc[len(df)] = [row_data[key] if key in row_data else None for key in columns]
        else:
            raise ValueError(f"No experiments results were found in {results_folder}")
        return df

    @classmethod
    def run(cls, args: List[str]) -> None:
        entry_point_utils = cls.entry_point_utils
        search_utils = cls.search_utils
        # A CUDA error message is issued when changing CUDA_VISIBLE_DEVICES
        # if processes are started in fork mode
        multiprocessing.set_start_method('spawn')
        # Parse benchmark arguments
        script_args = search_utils.script_parser().parse_args(args)
        # Retrieve the argument parser for the entrypoint
        entrypoint_parser = entry_point_utils.argument_parser
        # Instantiate directory for storing the results
        if not script_args.dry_run and not os.path.exists(script_args.results_folder):
            os.makedirs(script_args.results_folder)
        # If a benchmark YAML is passed, use that to retrieve argument combinations,
        # otherwise generate all possible combinations of arguments from the
        # entrypoint_parser
        args_dict = cls.standardize_args(script_args)
        # Generate a list of experiments
        exp_queue = search_utils.gen_search_space(
            args_dict, script_args, entrypoint_parser, entry_point_utils.validate)
        # Show a summary of the configuration to be run in the benchmark execution
        search_utils.print_benchmark_summary(args_dict, exp_queue, script_args, entrypoint_parser)
        # In the case of a dry-run, just stop after the output of the benchmark summary
        if script_args.dry_run:
            exit()
        # Prepare the shared queue for the processes
        args_queue = Queue()
        for args_tuple in exp_queue:
            args_queue.put(args_tuple)
        # Map the comma-separated string of GPU ids to a list
        cuda_available_devices = list(map(int, script_args.gpus.split(",")))
        # Number of argument combinations
        num_processes = len(cuda_available_devices) // script_args.num_gpus_per_process
        # Instantiate processes to run the argument combinations
        processes = []
        for i in range(num_processes):
            cuda_visible_devices = ",".join(
                map(str, cuda_available_devices[i:i + script_args.num_gpus_per_process]))
            process = multiprocessing.Process(
                target=run_args_bucket_process,
                args=(
                    entry_point_utils.entrypoint_main,
                    i,
                    num_processes,
                    cuda_visible_devices,
                    script_args.results_folder,
                    script_args.max_num_retries,
                    args_queue,
                ),
            )
            process.start()
            processes.append(process)

        # Wait for all processes to complete
        for process in processes:
            process.join()
        # Parse results
        df = cls.parse_results(script_args.results_folder)
        df.to_csv(f"{script_args.results_folder}/results.csv", index=False)
