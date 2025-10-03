import sys

from brevitas_examples.common.benchmark.utils import benchmark
from brevitas_examples.llm.benchmark.llm_benchmark import LLMBenchmarkUtils


class ExpansionBenchmark(LLMBenchmarkUtils):

    @staticmethod
    def validate(args, extra_args=None):
        super(LLMBenchmarkUtils, ExpansionBenchmark).validate(args, extra_args)
        if len(args.rotation_layers_to_expand) == 0:
            assert args.expansion_step == 0
        else:
            assert args.expansion_step != 0


if __name__ == "__main__":
    benchmark(ExpansionBenchmark, sys.argv[1:])
