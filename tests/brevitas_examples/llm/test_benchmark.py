# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from argparse import Namespace
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import patch

import pytest
import yaml

from brevitas_examples.common.benchmark.utils import GridSearchUtils
from brevitas_examples.common.benchmark.utils import RandomSearchUtils
from brevitas_examples.llm.benchmark.llm_benchmark import LLMEntryPointUtils
from brevitas_examples.llm.benchmark.llm_benchmark import LLMGridBenchmark
from brevitas_examples.llm.benchmark.llm_rand_benchmark import LLMRandomBenchmark
from brevitas_examples.llm.metric_report import FLOAT_PPL
from brevitas_examples.llm.metric_report import LogMetric
from brevitas_examples.llm.metric_report import QUANT_EAR
from brevitas_examples.llm.metric_report import QUANT_KLD
from brevitas_examples.llm.metric_report import QUANT_PPL
from brevitas_examples.llm.metric_report import result_keys
from tests.brevitas_examples.common import MockProcess
from tests.marker import skip_on_macos_nox

pytestmark = pytest.mark.llm

# ---------------------------------------------------------------------------
# Paths to test YAML configs (shipped alongside this test file)
# ---------------------------------------------------------------------------

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_GRID_YAML = os.path.join(_TEST_DIR, "benchmark_test_grid.yaml")
_RAND_YAML = os.path.join(_TEST_DIR, "benchmark_test_rand.yaml")

# Paths to the real template YAML files
_SRC_BENCHMARK_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "src",
    "brevitas_examples",
    "llm",
    "benchmark",
)
_REAL_GRID_YAML = os.path.join(_SRC_BENCHMARK_DIR, "benchmark_template.yaml")
_REAL_RAND_YAML = os.path.join(_SRC_BENCHMARK_DIR, "benchmark_rand_template.yaml")

# ---------------------------------------------------------------------------
# Mock entrypoint: validates basic args, returns dummy results
# ---------------------------------------------------------------------------


def _mock_entrypoint_main(
        args: Namespace,
        extra_args: Optional[List[str]] = None,
        job_folder: Optional[str] = None) -> Tuple[Dict, Any]:
    """Mock entrypoint that checks key args are present and returns dummy metrics."""
    # Validate that essential args exist
    assert hasattr(args, "model"), "args must have 'model'"
    assert hasattr(args, "weight_bit_width"), "args must have 'weight_bit_width'"
    return {"float_ppl": 10.0, "quant_ppl": 15.0, "quant_ear": 0.9, "quant_kld": 0.01}, None


# ====================== LLMEntryPointUtils: parse_log =====================


class TestLLMEntryPointUtils:

    def test_parse_log_float_and_quant_ppl(self):
        log = (
            "Loading model...\n"
            "Float perplexity (wikitext2): 25.123\n"
            "Running quantization...\n"
            "Quantized perplexity (wikitext2): 30.456\n"
            "Quantized expected acceptance rate (wikitext2): 0.987654\n"
            "Quantized KL divergence (wikitext2): 0.012345\n")
        result = LLMEntryPointUtils.parse_log(log)
        assert result["float_ppl"] == pytest.approx(25.123)
        assert result["quant_ppl"] == pytest.approx(30.456)
        assert result["quant_ear"] == pytest.approx(0.987654)
        assert result["quant_kld"] == pytest.approx(0.012345)
        # No standard error in the log -> the std fields are None
        assert result["float_ppl_std"] is None
        assert result["quant_ppl_std"] is None
        assert result["quant_ear_std"] is None
        assert result["quant_kld_std"] is None

    def test_parse_log_metrics_with_standard_error(self):
        log = (
            "Loading model...\n"
            "Float perplexity (wikitext2): 25.123 +/- 0.456\n"
            "Running quantization...\n"
            "Quantized perplexity (wikitext2): 30.456 +/- 1.234\n"
            "Quantized expected acceptance rate (wikitext2): 0.987654 +/- 0.001234\n"
            "Quantized KL divergence (wikitext2): -0.012345 +/- 0.000456\n")
        result = LLMEntryPointUtils.parse_log(log)
        assert result["float_ppl"] == pytest.approx(25.123)
        assert result["float_ppl_std"] == pytest.approx(0.456)
        assert result["quant_ppl"] == pytest.approx(30.456)
        assert result["quant_ppl_std"] == pytest.approx(1.234)
        assert result["quant_ear"] == pytest.approx(0.987654)
        assert result["quant_ear_std"] == pytest.approx(0.001234)
        assert result["quant_kld"] == pytest.approx(-0.012345)
        assert result["quant_kld_std"] == pytest.approx(0.000456)

    def test_parse_log_mixed_standard_error(self):
        # Only the float line carries a standard error; the quant line does not.
        log = (
            "Float perplexity (wikitext2): 25.123 +/- 0.456\n"
            "Quantized perplexity (wikitext2): 30.456\n")
        result = LLMEntryPointUtils.parse_log(log)
        assert result["float_ppl"] == pytest.approx(25.123)
        assert result["float_ppl_std"] == pytest.approx(0.456)
        assert result["quant_ppl"] == pytest.approx(30.456)
        assert result["quant_ppl_std"] is None

    def test_parse_log_missing_ppl(self):
        log = "Loading model...\nDone.\n"
        result = LLMEntryPointUtils.parse_log(log)
        assert result["float_ppl"] is None
        assert result["quant_ppl"] is None
        assert result["quant_ear"] is None
        assert result["quant_kld"] is None
        assert result["float_ppl_std"] is None
        assert result["quant_ppl_std"] is None
        assert result["quant_ear_std"] is None
        assert result["quant_kld_std"] is None

    def test_parse_log_with_few_shot_dict(self):
        log = (
            "Float perplexity (wikitext2): 25.0\n"
            "Quantized perplexity (wikitext2): 30.0\n"
            "Few-shot results: {'task_a': 0.85, 'task_b': 0.72}\n")
        result = LLMEntryPointUtils.parse_log(log)
        assert result["float_ppl"] == pytest.approx(25.0)
        assert result["quant_ppl"] == pytest.approx(30.0)
        assert result["task_a"] == pytest.approx(0.85)
        assert result["task_b"] == pytest.approx(0.72)

    def test_parse_log_standard_error_with_few_shot_dict(self):
        log = (
            "Float perplexity (wikitext2): 25.0 +/- 0.5\n"
            "Quantized perplexity (wikitext2): 30.0 +/- 0.75\n"
            "Few-shot results: {'task_a': 0.85, 'task_b': 0.72}\n")
        result = LLMEntryPointUtils.parse_log(log)
        assert result["float_ppl"] == pytest.approx(25.0)
        assert result["float_ppl_std"] == pytest.approx(0.5)
        assert result["quant_ppl"] == pytest.approx(30.0)
        assert result["quant_ppl_std"] == pytest.approx(0.75)
        assert result["task_a"] == pytest.approx(0.85)
        assert result["task_b"] == pytest.approx(0.72)

    def test_eval_metrics_derived_from_log_metrics(self):
        # The order defines the benchmark result column order, so it is pinned.
        assert LLMEntryPointUtils.eval_metrics == [
            "float_ppl",
            "float_ppl_std",
            "quant_ppl",
            "quant_ppl_std",
            "quant_ear",
            "quant_ear_std",
            "quant_kld",
            "quant_kld_std"]
        # A log without metrics still reports every declared key.
        assert list(LLMEntryPointUtils.parse_log("")) == LLMEntryPointUtils.eval_metrics

    def test_parse_log_round_trips_reported_metrics(self):
        # Guards the entrypoint (producer) against the benchmark (consumer): a
        # label changed on one side alone would break this.
        measured = {
            FLOAT_PPL: (25.123, 0.456),
            QUANT_PPL: (30.456, 1.234),
            QUANT_EAR: (0.987654, 0.001234),
            QUANT_KLD: (-0.012345, 0.000456),}
        log = "\n".join(
            metric.report("wikitext2", value, std) for metric, (value, std) in measured.items())

        result = LLMEntryPointUtils.parse_log(log)

        for metric, (value, standard_error) in measured.items():
            assert result[metric.name] == pytest.approx(value, abs=10 ** -metric.precision)
            assert result[metric.std_name] == pytest.approx(
                standard_error, abs=10 ** -metric.precision)

    def test_subclass_can_extend_log_metrics(self):

        class ExtendedEntryPointUtils(LLMEntryPointUtils):
            log_metrics = LLMEntryPointUtils.log_metrics + (
                LogMetric("quant_extra", "Quantized extra metric"),)
            eval_metrics = result_keys(log_metrics)

        assert ExtendedEntryPointUtils.eval_metrics[-2:] == ["quant_extra", "quant_extra_std"]
        result = ExtendedEntryPointUtils.parse_log(
            "Float perplexity (wikitext2): 25.0\n"
            "Quantized extra metric (wikitext2): 1.5 +/- 0.25\n")
        assert result["float_ppl"] == pytest.approx(25.0)
        assert result["quant_extra"] == pytest.approx(1.5)
        assert result["quant_extra_std"] == pytest.approx(0.25)
        # The base class is unaffected by the subclass' extra metric.
        assert "quant_extra" not in LLMEntryPointUtils.eval_metrics

    def test_validate_valid_default_args(self):
        """Default args from the parser should be valid."""
        parser = LLMEntryPointUtils.argument_parser
        default_args = parser.parse_args([])
        # Should not raise
        LLMEntryPointUtils.validate(default_args)

    def test_validate_invalid_gptq_and_gpfq(self):
        """Enabling GPTQ and GPFQ together should fail validation."""
        parser = LLMEntryPointUtils.argument_parser
        default_args = parser.parse_args([])
        default_args.gptq = True
        default_args.gpfq = True
        default_args.no_quantize = False
        with pytest.raises(AssertionError):
            LLMEntryPointUtils.validate(default_args)


# ==================== LLMGridBenchmark (Grid Search) ======================


class TestLLMGridBenchmark:

    def test_standardize_args_from_test_yaml(self):
        script_args = Namespace(config=_GRID_YAML, results_folder="./")
        args_dict = LLMGridBenchmark.standardize_args(script_args)
        # Keys from the YAML should be preserved as lists
        assert args_dict["model"] == ["facebook/opt-125m"]
        assert args_dict["weight_bit_width"] == [4, 8]
        assert args_dict["dataset"] == ["wikitext2"]
        # Missing keys should be filled from parser defaults (as single-element lists)
        assert isinstance(args_dict.get("weight_quant_granularity"), list)

    @pytest.mark.skipif(
        not os.path.exists(_REAL_GRID_YAML), reason="Real benchmark_template.yaml not found")
    def test_standardize_args_from_real_template(self):
        script_args = Namespace(config=_REAL_GRID_YAML, results_folder="./")
        args_dict = LLMGridBenchmark.standardize_args(script_args)
        # Every value should be a list
        for key, value in args_dict.items():
            assert isinstance(value, list), f"Key '{key}' should be a list, got {type(value)}"

    def test_gen_search_space_small_config(self):
        script_args = Namespace(config=_GRID_YAML, results_folder="./")
        args_dict = LLMGridBenchmark.standardize_args(script_args)
        grid_args = Namespace(start_index=0, end_index=-1, shuffle_seed=None)
        exp_queue = GridSearchUtils.gen_search_space(
            args_dict, grid_args, LLMEntryPointUtils.argument_parser, LLMEntryPointUtils.validate)
        # 1 model * 2 bit_widths * (all other single-value defaults) = 2 combos
        # Some combos might be filtered by validate, but with simple defaults most pass
        assert len(exp_queue) >= 1
        for args, extra_args, full_dict in exp_queue:
            assert hasattr(args, "model")
            assert hasattr(args, "weight_bit_width")
            assert args.model == "facebook/opt-125m"
            assert args.weight_bit_width in [4, 8]

    @skip_on_macos_nox
    def test_benchmark_e2e(self, tmp_path):
        results_folder = str(tmp_path / "results")

        with patch.object(LLMEntryPointUtils, "entrypoint_main", _mock_entrypoint_main):
            with patch("brevitas_examples.common.benchmark.utils.multiprocessing.Process",
                       MockProcess):
                with patch("brevitas_examples.common.benchmark.utils."
                           "multiprocessing.set_start_method",
                           lambda *a,
                           **kw: None):
                    LLMGridBenchmark.run([
                        "--config",
                        _GRID_YAML,
                        "--results-folder",
                        results_folder,
                        "--gpus",
                        "0",],)

        csv_path = os.path.join(results_folder, "results.csv")
        assert os.path.exists(csv_path)
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Should have at least 1 row (depends on validation filtering)
        assert len(df) >= 1
        assert "float_ppl" in df.columns
        assert "quant_ppl" in df.columns
        assert "quant_ear" in df.columns
        assert "quant_kld" in df.columns
        assert set(df["status"]) == {"successful"}
        # Verify the mock entrypoint results are in the CSV
        assert all(df["float_ppl"] == 10.0)
        assert all(df["quant_ppl"] == 15.0)
        assert all(df["quant_ear"] == 0.9)
        assert all(df["quant_kld"] == 0.01)


# ================= LLMRandomBenchmark (Random Search) =====================


class TestLLMRandomBenchmark:

    def test_standardize_args_from_test_yaml(self):
        script_args = Namespace(config=_RAND_YAML, results_folder="./")
        args_dict = LLMRandomBenchmark.standardize_args(script_args)
        # Keys from the YAML should be preserved as dicts with rand_type/rand_values
        assert args_dict["model"]["rand_type"] == "const"
        assert args_dict["model"]["rand_values"] == "facebook/opt-125m"
        assert args_dict["weight_bit_width"]["rand_type"] == "choices"
        assert args_dict["weight_bit_width"]["rand_values"] == [4, 8]
        # Missing keys should be filled
        assert "weight_quant_granularity" in args_dict
        assert args_dict["weight_quant_granularity"]["rand_type"] == "const"

    @pytest.mark.skipif(
        not os.path.exists(_REAL_RAND_YAML), reason="Real benchmark_rand_template.yaml not found")
    def test_standardize_args_from_real_template(self):
        script_args = Namespace(config=_REAL_RAND_YAML, results_folder="./")
        args_dict = LLMRandomBenchmark.standardize_args(script_args)
        # Every value should be a dict with rand_type and rand_values
        for key, value in args_dict.items():
            assert isinstance(value, dict), f"Key '{key}' should be a dict, got {type(value)}"
            assert "rand_type" in value, f"Key '{key}' missing 'rand_type'"
            assert "rand_values" in value, f"Key '{key}' missing 'rand_values'"

    def test_gen_search_space_small_config(self):
        script_args = Namespace(config=_RAND_YAML, results_folder="./")
        args_dict = LLMRandomBenchmark.standardize_args(script_args)
        search_args = Namespace(
            num_experiments=3,
            max_experimental_configs=1000,
            seed=0,
            config=_RAND_YAML,
            results_folder="./",
        )
        exp_queue = RandomSearchUtils.gen_search_space(
            args_dict, search_args, LLMEntryPointUtils.argument_parser, LLMEntryPointUtils.validate)
        assert len(exp_queue) <= 3
        # Should get at least 1 valid experiment
        assert len(exp_queue) >= 1
        for args, extra_args, full_dict in exp_queue:
            assert hasattr(args, "model")
            assert args.model == "facebook/opt-125m"

    @skip_on_macos_nox
    def test_benchmark_e2e(self, tmp_path):
        results_folder = str(tmp_path / "results")

        with patch.object(LLMEntryPointUtils, "entrypoint_main", _mock_entrypoint_main):
            with patch("brevitas_examples.common.benchmark.utils.multiprocessing.Process",
                       MockProcess):
                with patch("brevitas_examples.common.benchmark.utils."
                           "multiprocessing.set_start_method",
                           lambda *a,
                           **kw: None):
                    LLMRandomBenchmark.run([
                        "--config",
                        _RAND_YAML,
                        "--results-folder",
                        results_folder,
                        "--gpus",
                        "0",
                        "--num-experiments",
                        "2",],)

        csv_path = os.path.join(results_folder, "results.csv")
        assert os.path.exists(csv_path)
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) >= 1
        assert "float_ppl" in df.columns
        assert "quant_ppl" in df.columns
        assert "quant_ear" in df.columns
        assert "quant_kld" in df.columns
        assert set(df["status"]) == {"successful"}
