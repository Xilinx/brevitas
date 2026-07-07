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
from tests.brevitas_examples.common import MockProcess
from tests.marker import skip_on_macos_nox

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
    return {"float_ppl": 10.0, "quant_ppl": 15.0}, None


# ====================== LLMEntryPointUtils: parse_log =====================


class TestLLMEntryPointUtils:

    @pytest.mark.llm
    def test_parse_log_float_and_quant_ppl(self):
        log = (
            "Loading model...\n"
            "Float perplexity (wikitext2): 25.123\n"
            "Running quantization...\n"
            "Quantized perplexity (wikitext2): 30.456\n")
        result = LLMEntryPointUtils.parse_log(log)
        assert result["float_ppl"] == pytest.approx(25.123)
        assert result["quant_ppl"] == pytest.approx(30.456)

    @pytest.mark.llm
    def test_parse_log_missing_ppl(self):
        log = "Loading model...\nDone.\n"
        result = LLMEntryPointUtils.parse_log(log)
        assert result["float_ppl"] is None
        assert result["quant_ppl"] is None

    @pytest.mark.llm
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

    @pytest.mark.llm
    def test_validate_valid_default_args(self):
        """Default args from the parser should be valid."""
        parser = LLMEntryPointUtils.argument_parser
        default_args = parser.parse_args([])
        # Should not raise
        LLMEntryPointUtils.validate(default_args)

    @pytest.mark.llm
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

    @pytest.mark.llm
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
    @pytest.mark.llm
    def test_standardize_args_from_real_template(self):
        script_args = Namespace(config=_REAL_GRID_YAML, results_folder="./")
        args_dict = LLMGridBenchmark.standardize_args(script_args)
        # Every value should be a list
        for key, value in args_dict.items():
            assert isinstance(value, list), f"Key '{key}' should be a list, got {type(value)}"

    @pytest.mark.llm
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

    @pytest.mark.llm
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
        assert set(df["status"]) == {"successful"}
        # Verify the mock entrypoint results are in the CSV
        assert all(df["float_ppl"] == 10.0)
        assert all(df["quant_ppl"] == 15.0)


# ================= LLMRandomBenchmark (Random Search) =====================


class TestLLMRandomBenchmark:

    @pytest.mark.llm
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
    @pytest.mark.llm
    def test_standardize_args_from_real_template(self):
        script_args = Namespace(config=_REAL_RAND_YAML, results_folder="./")
        args_dict = LLMRandomBenchmark.standardize_args(script_args)
        # Every value should be a dict with rand_type and rand_values
        for key, value in args_dict.items():
            assert isinstance(value, dict), f"Key '{key}' should be a dict, got {type(value)}"
            assert "rand_type" in value, f"Key '{key}' missing 'rand_type'"
            assert "rand_values" in value, f"Key '{key}' missing 'rand_values'"

    @pytest.mark.llm
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

    @pytest.mark.llm
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
        assert set(df["status"]) == {"successful"}
