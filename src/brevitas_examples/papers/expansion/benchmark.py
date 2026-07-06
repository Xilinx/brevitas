import sys

from brevitas_examples.common.benchmark.utils import BenchmarkUtils
from brevitas_examples.common.benchmark.utils import GridSearchUtils
from brevitas_examples.llm.benchmark.llm_benchmark import LLMEntryPointUtils


class ExpansionEntryPointUtils(LLMEntryPointUtils):

    @staticmethod
    def validate(args, extra_args=None):
        LLMEntryPointUtils.validate(args, extra_args)
        if len(args.rotation_layers_to_expand) == 0:
            assert args.expansion_step == 0
        else:
            assert args.expansion_step != 0


class ExpansionBenchmark(BenchmarkUtils):
    entry_point_utils = ExpansionEntryPointUtils
    search_utils = GridSearchUtils


if __name__ == "__main__":
    ExpansionBenchmark.run(sys.argv[1:])
