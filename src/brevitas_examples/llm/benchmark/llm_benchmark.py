from argparse import ArgumentParser
from argparse import Namespace
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.llm.llm_args import create_llm_args_parser
from brevitas_examples.llm.llm_args import validate as validate_llm_args


class LLMBenchmarkUtils(BenchmarkUtils):

    argument_parser: ArgumentParser = create_llm_args_parser()
    eval_metrics: List[str] = ["float_ppl", "quant_ppl"]

    @staticmethod
    def parse_log(job_log: str) -> Dict[str, Any]:
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
        # Return the results from the log as a dictionary
        job_log_results = {
            "float_ppl": float_ppl,
            "quant_ppl": quant_ppl,
            **few_shot_eval,}
        return job_log_results

    @staticmethod
    def validate(args: Namespace, extra_args: Optional[List[str]] = None) -> None:
        validate_llm_args(args=args, extra_args=extra_args)

    @staticmethod
    def entrypoint_main(args: Namespace,
                        extra_args: Optional[List[str]] = None) -> Tuple[Dict, Any]:
        from brevitas_examples.llm.main import quantize_llm
        return quantize_llm(args=args, extra_args=extra_args)


if __name__ == "__main__":
    benchmark(LLMBenchmarkUtils, sys.argv[1:])
