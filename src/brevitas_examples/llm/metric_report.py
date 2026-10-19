# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Single source of truth for how evaluation metrics are written to, and read
back from, a job log.

The entrypoint reports metrics with :meth:`LogMetric.report`, and the benchmark
recovers them from a crashed job's log with :meth:`LogMetric.parse`. Keeping
both sides here means a label can never drift out of sync with the pattern that
reads it.

This module is intentionally free of heavy imports: the benchmark entry points
import it without pulling in torch.
"""

import re
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple


class LogMetric(NamedTuple):
    """A metric reported as ``<label> (<dataset>): <value> [+/- <standard error>]``."""

    name: str
    label: str
    precision: int = 3

    @property
    def std_name(self) -> str:
        return f"{self.name}_std"

    def format(self, value: float, standard_error: Optional[float]) -> str:
        # Reported as "<value> +/- <standard error>". ASCII is used on purpose,
        # since the logs are also read back on Windows consoles.
        if standard_error is None:
            return f"{value:.{self.precision}f}"
        return f"{value:.{self.precision}f} +/- {standard_error:.{self.precision}f}"

    def report(self, dataset: str, value: float, standard_error: Optional[float]) -> str:
        return f"{self.label} ({dataset}): {self.format(value, standard_error)}"

    def parse(self, job_log: str) -> Tuple[Optional[float], Optional[float]]:
        # The standard error group is optional so that logs produced before it
        # was reported still parse. A standard error is never negative.
        match = re.search(
            rf"{re.escape(self.label)} \((.*?)\): (-?\d+\.\d+)(?:\s*\+/-\s*(\d+\.\d+))?", job_log)
        if match is None:
            return None, None
        standard_error = float(match.group(3)) if match.group(3) is not None else None
        return float(match.group(2)), standard_error


FLOAT_PPL = LogMetric("float_ppl", "Float perplexity")
QUANT_PPL = LogMetric("quant_ppl", "Quantized perplexity")
QUANT_EAR = LogMetric("quant_ear", "Quantized expected acceptance rate", precision=6)
QUANT_KLD = LogMetric("quant_kld", "Quantized KL divergence", precision=6)

LLM_LOG_METRICS: Tuple[LogMetric, ...] = (FLOAT_PPL, QUANT_PPL, QUANT_EAR, QUANT_KLD)


def result_keys(metrics: Iterable[LogMetric]) -> List[str]:
    """Return the result dictionary keys contributed by ``metrics``, in order."""
    return [key for metric in metrics for key in (metric.name, metric.std_name)]
