# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import ArgumentParser
from argparse import Namespace
from multiprocessing import Queue
import os
import random
from types import SimpleNamespace
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import patch

import pytest
import yaml

from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.common.benchmark.utils import ConstNode
from brevitas_examples.common.benchmark.utils import EntryPointUtils
from brevitas_examples.common.benchmark.utils import GridSearchUtils
from brevitas_examples.common.benchmark.utils import RandomArgNode
from brevitas_examples.common.benchmark.utils import RandomSearchUtils
from brevitas_examples.common.benchmark.utils import run_args_bucket_process
from tests.brevitas_examples.common import MockProcess
from tests.marker import skip_on_macos_nox

# ---------------------------------------------------------------------------
# Mock utilities: a minimal BenchmarkUtils with a 3-argument parser
# ---------------------------------------------------------------------------


def _create_mock_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Mock entrypoint")
    parser.add_argument("--model", type=str, default="model_a", choices=["model_a", "model_b"])
    parser.add_argument("--bit-width", type=int, default=8, choices=[4, 8])
    parser.add_argument("--method", type=str, default="ptq", choices=["ptq", "qat"])
    return parser


def _mock_validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
    """Reject ptq + 4-bit as invalid for testing purposes."""
    if getattr(args, "method", None) == "ptq" and getattr(args, "bit_width", None) == 4:
        raise AssertionError("qat is required with 4-bit")


def _mock_entrypoint_main(
        args: Namespace,
        extra_args: Optional[List[str]] = None,
        job_folder: Optional[str] = None) -> Tuple[Dict, Any]:
    return {"metric_a": 1.0, "metric_b": 2.0}, None


class MockEntryPointUtils(EntryPointUtils):
    argument_parser: ArgumentParser = _create_mock_parser()
    eval_metrics: List[str] = ["metric_a", "metric_b"]

    @staticmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
        return {}

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        _mock_validate(args, extra_args)

    @staticmethod
    def entrypoint_main(
            args: Namespace,
            extra_args: Optional[List[str]] = None,
            job_folder: Optional[str] = None) -> Tuple[Dict, Any]:
        return _mock_entrypoint_main(args, extra_args, job_folder)


class MockGridBenchmark(BenchmarkUtils):
    entry_point_utils = MockEntryPointUtils
    search_utils = GridSearchUtils


class MockRandomBenchmark(BenchmarkUtils):
    entry_point_utils = MockEntryPointUtils
    search_utils = RandomSearchUtils


# ============================= RandomArgNode ==============================


class TestRandomArgNode:

    def test_const_value(self):
        node = RandomArgNode.from_config(rand_type="const", rand_values=42)
        assert node.value() == 42

    def test_choices_deterministic(self):
        """Same seed produces same sequence of choices."""
        node = RandomArgNode.from_config(rand_type="choices", rand_values=[1, 2, 3, 4, 5])
        random.seed(42)
        values_a = [node.value() for _ in range(10)]
        random.seed(42)
        values_b = [node.value() for _ in range(10)]
        assert values_a == values_b

    def test_linear_value_in_range(self):
        random.seed(0)
        node = RandomArgNode.from_config(rand_type="linear", rand_values=[0.0, 1.0])
        for _ in range(20):
            v = node.value()
            assert 0.0 <= v <= 1.0

    def test_log2_value_in_range(self):
        random.seed(0)
        node = RandomArgNode.from_config(rand_type="log2", rand_values=[1.0, 16.0])
        for _ in range(20):
            v = node.value()
            assert 1.0 <= v <= 16.0

    def test_exp2_value_in_range(self):
        random.seed(0)
        node = RandomArgNode.from_config(rand_type="exp2", rand_values=[0.0, 4.0])
        for _ in range(20):
            v = node.value()
            assert 0.0 <= v <= 4.0

    def test_invalid_rand_type(self):
        # Unknown types are rejected at construction time (fail fast).
        with pytest.raises(ValueError, match="not a valid random type"):
            RandomArgNode.from_config(rand_type="bad_type", rand_values=[1, 2])

    def test_range_requires_pair(self):
        # Range types need exactly [min, max].
        with pytest.raises(ValueError, match="min, max"):
            RandomArgNode.from_config(rand_type="linear", rand_values=[1.0])

    def test_range_requires_ordered_bounds(self):
        with pytest.raises(ValueError, match="min <= max"):
            RandomArgNode.from_config(rand_type="linear", rand_values=[1.0, 0.0])

    def test_range_requires_numeric_bounds(self):
        with pytest.raises(ValueError, match="numeric"):
            RandomArgNode.from_config(rand_type="linear", rand_values=["a", "b"])

    def test_log2_requires_positive_bounds(self):
        with pytest.raises(ValueError, match="strictly positive"):
            RandomArgNode.from_config(rand_type="log2", rand_values=[0.0, 16.0])

    def test_missing_rand_type_fails_at_instantiation(self):
        # A concrete node that forgets to override `rand_type` stays abstract:
        # it can be defined, but ABC rejects instantiation.
        class NoRandType(RandomArgNode):

            def value(self):
                return 1

        # It must not have been registered.
        assert NoRandType not in RandomArgNode._registry.values()
        with pytest.raises(TypeError, match="rand_type"):
            NoRandType(rand_values=1)

    def test_non_str_rand_type_fails_at_definition(self):
        with pytest.raises(TypeError, match="must be a str"):

            class NonStrRandType(RandomArgNode):
                rand_type = 5

                def value(self):
                    return 1

    def test_duplicate_rand_type_fails_at_definition(self):
        # Subclassing a leaf without a new `rand_type` reuses an existing key.
        with pytest.raises(TypeError, match="already registered"):

            class DuplicateConst(ConstNode):
                pass


# ============================= GridSearchUtils ============================


class TestGridSearchUtils:

    def test_standardize_args_from_yaml(self, tmp_path):
        config = {
            "model": ["model_a", "model_b"],
            "bit_width": [4, 8],}
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)
        script_args = Namespace(config=yaml_path, results_folder=str(tmp_path))
        args_dict = MockGridBenchmark.standardize_args(script_args)
        # Provided keys should be preserved
        assert args_dict["model"] == ["model_a", "model_b"]
        assert args_dict["bit_width"] == [4, 8]
        # Missing keys should be filled with the default value (as single-element list)
        assert "method" in args_dict
        assert args_dict["method"] == ["ptq"]

    def test_standardize_args_no_config_raises(self):
        script_args = Namespace(config=None, results_folder="./")
        with pytest.raises(ValueError, match="Config file not specified"):
            MockGridBenchmark.standardize_args(script_args)

    def test_gen_search_space_cartesian_product(self):
        args_dict = {
            "model": ["model_a", "model_b"],
            "bit_width": [4, 8],
            "method": ["qat"],}
        script_args = Namespace(start_index=0, end_index=-1, shuffle_seed=None)
        exp_queue = GridSearchUtils.gen_search_space(
            args_dict,
            script_args,
            MockEntryPointUtils.argument_parser,
            MockEntryPointUtils.validate)
        # 2 models * 2 bit_widths * 1 method = 4 combos, minus invalid (qat+4-bit) = 4
        assert len(exp_queue) == 4
        # Each entry is (args, extra_args, args_dict)
        for args, extra_args, full_dict in exp_queue:
            assert hasattr(args, "model")
            assert hasattr(args, "bit_width")
            assert isinstance(extra_args, list)
            assert isinstance(full_dict, dict)

    def test_gen_search_space_validation_filtering(self):
        args_dict = {
            "model": ["model_a"],
            "bit_width": [4, 8],
            "method": ["ptq", "qat"],}
        script_args = Namespace(start_index=0, end_index=-1, shuffle_seed=None)
        exp_queue = GridSearchUtils.gen_search_space(
            args_dict,
            script_args,
            MockEntryPointUtils.argument_parser,
            MockEntryPointUtils.validate)
        # 1 * 2 * 2 = 4 total, but ptq+4-bit is invalid => 3 valid
        assert len(exp_queue) == 3
        for args, _, _ in exp_queue:
            # Ensure the invalid combo was filtered
            assert not (args.method == "ptq" and args.bit_width == 4)

    def test_gen_search_space_start_end_index(self):
        args_dict = {
            "model": ["model_a", "model_b"],
            "bit_width": [4, 8],
            "method": ["qat"],}
        script_args = Namespace(start_index=1, end_index=3, shuffle_seed=None)
        exp_queue = GridSearchUtils.gen_search_space(
            args_dict,
            script_args,
            MockEntryPointUtils.argument_parser,
            MockEntryPointUtils.validate)
        # Slice [1:3] of 4 valid combos = 2
        assert len(exp_queue) == 2

    def test_gen_search_space_shuffle_deterministic(self):
        args_dict = {
            "model": ["model_a", "model_b"],
            "bit_width": [4, 8],
            "method": ["qat"],}
        script_args = Namespace(start_index=0, end_index=-1, shuffle_seed=123)
        exp_a = GridSearchUtils.gen_search_space(
            args_dict,
            script_args,
            MockEntryPointUtils.argument_parser,
            MockEntryPointUtils.validate)
        exp_b = GridSearchUtils.gen_search_space(
            args_dict,
            script_args,
            MockEntryPointUtils.argument_parser,
            MockEntryPointUtils.validate)
        # Same seed should produce same order
        for (a_args, _, a_dict), (b_args, _, b_dict) in zip(exp_a, exp_b):
            assert a_dict == b_dict

    def test_gen_search_space_extra_args(self):
        """Keys not in the parser should be passed as extra_args."""
        args_dict = {
            "model": ["model_a"],
            "bit_width": [8],
            "method": ["ptq"],
            "unknown_param": ["value1"],}
        script_args = Namespace(start_index=0, end_index=-1, shuffle_seed=None)
        # Validation rejects extra_args for the mock, so we patch validate to allow it
        with patch.object(MockEntryPointUtils, 'validate', lambda *a, **kw: None):
            exp_queue = GridSearchUtils.gen_search_space(
                args_dict,
                script_args,
                MockEntryPointUtils.argument_parser,
                MockEntryPointUtils.validate)
        assert len(exp_queue) == 1
        _, extra_args, _ = exp_queue[0]
        assert "--unknown-param" in extra_args
        assert "value1" in extra_args


# ============================ RandomSearchUtils ===========================


class TestRandomSearchUtils:

    def test_standardize_args_from_yaml(self, tmp_path):
        config = {
            "model": {
                "rand_type": "choices", "rand_values": ["model_a", "model_b"]},
            "bit_width": {
                "rand_type": "const", "rand_values": 8},}
        yaml_path = str(tmp_path / "config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)
        script_args = Namespace(config=yaml_path, results_folder=str(tmp_path))
        args_dict = MockRandomBenchmark.standardize_args(script_args)
        # Provided keys preserved
        assert args_dict["model"]["rand_type"] == "choices"
        assert args_dict["bit_width"]["rand_type"] == "const"
        # Missing keys should be filled with const default value
        assert "method" in args_dict
        assert args_dict["method"]["rand_type"] == "const"
        assert args_dict["method"]["rand_values"] == "ptq"

    def test_gen_search_space_num_experiments(self):
        args_dict = {
            "model": {
                "rand_type": "const", "rand_values": "model_a"},
            "bit_width": {
                "rand_type": "choices", "rand_values": [4, 8]},
            "method": {
                "rand_type": "const", "rand_values": "ptq"},}
        script_args = Namespace(
            num_experiments=5,
            max_experimental_configs=1000,
            seed=0,
            config=None,
            results_folder="./",
        )
        exp_queue = RandomSearchUtils.gen_search_space(
            args_dict,
            script_args,
            MockEntryPointUtils.argument_parser,
            MockEntryPointUtils.validate)
        assert len(exp_queue) == 5

    def test_gen_search_space_validation_filtering(self):
        """With only invalid combos possible, should produce fewer than requested."""
        args_dict = {
            "model": {
                "rand_type": "const", "rand_values": "model_a"},
            "bit_width": {
                "rand_type": "const", "rand_values": 4},
            "method": {
                "rand_type": "const", "rand_values": "ptq"},}
        script_args = Namespace(
            num_experiments=5,
            max_experimental_configs=20,
            seed=0,
            config=None,
            results_folder="./",
        )
        exp_queue = RandomSearchUtils.gen_search_space(
            args_dict,
            script_args,
            MockEntryPointUtils.argument_parser,
            MockEntryPointUtils.validate)
        # ptq + 4-bit is always invalid, so no valid configs
        assert len(exp_queue) == 0

    def test_gen_search_space_deterministic_with_seed(self):
        args_dict = {
            "model": {
                "rand_type": "choices", "rand_values": ["model_a", "model_b"]},
            "bit_width": {
                "rand_type": "choices", "rand_values": [4, 8]},
            "method": {
                "rand_type": "const", "rand_values": "ptq"},}
        script_args = Namespace(
            num_experiments=3,
            max_experimental_configs=1000,
            seed=0,
            config=None,
            results_folder="./",
        )
        # gen_search_space does not seed internally -- it relies on caller
        # The run() method itself doesn't seed either for RandomSearchUtils;
        # we test that given the same initial random state, we get the same results.
        random.seed(42)
        exp_a = RandomSearchUtils.gen_search_space(
            args_dict,
            script_args,
            MockEntryPointUtils.argument_parser,
            MockEntryPointUtils.validate)
        random.seed(42)
        exp_b = RandomSearchUtils.gen_search_space(
            args_dict,
            script_args,
            MockEntryPointUtils.argument_parser,
            MockEntryPointUtils.validate)
        for (_, _, dict_a), (_, _, dict_b) in zip(exp_a, exp_b):
            assert dict_a == dict_b


# ======================== run_args_bucket_process =========================


class TestRunArgsBucketProcess:

    @staticmethod
    def _make_queue(items):
        q = Queue()
        for item in items:
            q.put(item)
        return q

    @skip_on_macos_nox
    def test_successful_run(self, tmp_path):
        """Entrypoint succeeds: config.yaml and run_results.yaml are created."""
        results_folder = str(tmp_path)
        args = SimpleNamespace(model="model_a", bit_width=8)
        args_dict = {"model": "model_a", "bit_width": 8}
        queue = self._make_queue([(args, [], args_dict)])

        run_args_bucket_process(
            main_entrypoint=_mock_entrypoint_main,
            id=0,
            num_processes=1,
            cuda_visible_devices="0",
            results_folder=results_folder,
            max_num_retries=1,
            args_queue=queue,
        )

        # Find the job folder (MD5 hash name)
        job_dirs = [
            d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
        assert len(job_dirs) == 1
        job_folder = os.path.join(results_folder, job_dirs[0])
        # Check config.yaml
        with open(os.path.join(job_folder, "config.yaml")) as f:
            saved_config = yaml.safe_load(f)
        assert saved_config["model"] == "model_a"
        assert saved_config["bit_width"] == 8
        # Check run_results.yaml
        with open(os.path.join(job_folder, "run_results.yaml")) as f:
            results = yaml.safe_load(f)
        assert results["status"] == "successful"
        assert results["metric_a"] == 1.0
        assert results["metric_b"] == 2.0
        assert "elapsed_time" in results

    @skip_on_macos_nox
    def test_crashed_run_retry(self, tmp_path):
        """Entrypoint crashes: retries and records crashed status."""
        results_folder = str(tmp_path)
        args = SimpleNamespace(model="model_a", bit_width=8)
        args_dict = {"model": "model_a", "bit_width": 8}
        queue = self._make_queue([(args, [], args_dict)])

        def crashing_entrypoint(args, extra_args=None, job_folder=None):
            raise RuntimeError("Simulated crash")

        run_args_bucket_process(
            main_entrypoint=crashing_entrypoint,
            id=0,
            num_processes=1,
            cuda_visible_devices="0",
            results_folder=results_folder,
            max_num_retries=2,
            args_queue=queue,
        )

        job_dirs = [
            d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
        assert len(job_dirs) == 1
        job_folder = os.path.join(results_folder, job_dirs[0])
        with open(os.path.join(job_folder, "run_results.yaml")) as f:
            results = yaml.safe_load(f)
        assert results["status"] == "crashed"
        assert results["retry_number"] == 2

    @skip_on_macos_nox
    def test_skip_existing_successful(self, tmp_path):
        """Pre-existing successful job is skipped."""
        import hashlib

        from brevitas_examples.common.benchmark.utils import _dict_to_bytes

        results_folder = str(tmp_path)
        args = SimpleNamespace(model="model_a", bit_width=8)
        args_dict = {"model": "model_a", "bit_width": 8}
        job_name = hashlib.md5(_dict_to_bytes(args_dict)).hexdigest()
        job_folder = os.path.join(results_folder, job_name)
        os.makedirs(job_folder)

        # Pre-populate run_results.yaml with successful status
        with open(os.path.join(job_folder, "run_results.yaml"), "w") as f:
            yaml.dump({"status": "successful", "metric_a": 99.0}, f)
        with open(os.path.join(job_folder, "config.yaml"), "w") as f:
            yaml.dump(args_dict, f)

        call_count = {"n": 0}

        def counting_entrypoint(args, extra_args=None, job_folder=None):
            call_count["n"] += 1
            return {"metric_a": 1.0, "metric_b": 2.0}, None

        queue = self._make_queue([(args, [], args_dict)])
        run_args_bucket_process(
            main_entrypoint=counting_entrypoint,
            id=0,
            num_processes=1,
            cuda_visible_devices="0",
            results_folder=results_folder,
            max_num_retries=1,
            args_queue=queue,
        )

        # Entrypoint should not have been called
        assert call_count["n"] == 0
        # Original results should be preserved
        with open(os.path.join(job_folder, "run_results.yaml")) as f:
            results = yaml.safe_load(f)
        assert results["metric_a"] == 99.0


# ========================= benchmark() orchestrator =======================


class TestBenchmarkOrchestrator:

    @staticmethod
    def _write_grid_yaml(tmp_path):
        config = {
            "model": ["model_a", "model_b"],
            "bit_width": [8],
            "method": ["ptq"],}
        yaml_path = str(tmp_path / "grid_config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)
        return yaml_path

    @skip_on_macos_nox
    def test_benchmark_grid_search_e2e(self, tmp_path):
        yaml_path = self._write_grid_yaml(tmp_path)
        results_folder = str(tmp_path / "results")

        with patch("brevitas_examples.common.benchmark.utils.multiprocessing.Process", MockProcess):
            with patch("brevitas_examples.common.benchmark.utils."
                       "multiprocessing.set_start_method",
                       lambda *a,
                       **kw: None):
                MockGridBenchmark.run([
                    "--config",
                    yaml_path,
                    "--results-folder",
                    results_folder,
                    "--gpus",
                    "0",],)

        # Check that results.csv was created
        csv_path = os.path.join(results_folder, "results.csv")
        assert os.path.exists(csv_path)
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "metric_a" in df.columns
        assert set(df["status"]) == {"successful"}

    def test_benchmark_dry_run(self, tmp_path):
        yaml_path = self._write_grid_yaml(tmp_path)
        results_folder = str(tmp_path / "results")

        with pytest.raises(SystemExit):
            with patch("brevitas_examples.common.benchmark.utils.multiprocessing.set_start_method",
                       lambda *a,
                       **kw: None):
                MockGridBenchmark.run([
                    "--config",
                    yaml_path,
                    "--results-folder",
                    results_folder,
                    "--dry-run",],)

        # results folder should not have been populated with CSV
        csv_path = os.path.join(results_folder, "results.csv")
        assert not os.path.exists(csv_path)

    @skip_on_macos_nox
    def test_benchmark_random_search_e2e(self, tmp_path):
        config = {
            "model": {
                "rand_type": "choices", "rand_values": ["model_a", "model_b"]},
            "bit_width": {
                "rand_type": "choices", "rand_values": [4, 8]},
            "method": {
                "rand_type": "const", "rand_values": "qat"},}
        yaml_path = str(tmp_path / "rand_config.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump(config, f)
        results_folder = str(tmp_path / "results")

        with patch("brevitas_examples.common.benchmark.utils.multiprocessing.Process", MockProcess):
            with patch("brevitas_examples.common.benchmark.utils."
                       "multiprocessing.set_start_method",
                       lambda *a,
                       **kw: None):
                MockRandomBenchmark.run([
                    "--config",
                    yaml_path,
                    "--results-folder",
                    results_folder,
                    "--gpus",
                    "0",
                    "--num-experiments",
                    "3",],)

        csv_path = os.path.join(results_folder, "results.csv")
        assert os.path.exists(csv_path)
        import pandas as pd
        df = pd.read_csv(csv_path)
        # 2 models * 2 bit_widths = 4 unique combos, requesting 3
        assert len(df) >= 2
        assert len(df) <= 3
        assert set(df["status"]) == {"successful"}
