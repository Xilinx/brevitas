import warnings

from brevitas.core.scaling import (
    ParameterFromRuntimeStatsScaling,
)
from brevitas.core.stats.stats_op import AbsMax


def test_scaling_state_dict():
    scaling_op = ParameterFromRuntimeStatsScaling(
        collect_stats_steps=10, scaling_stats_impl=AbsMax()
    )

    with warnings.catch_warnings(record=True) as wlist:
        scaling_op.state_dict()
        for w in wlist:
            assert "Positional args are being deprecated" not in str(w.message)
