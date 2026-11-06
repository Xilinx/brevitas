# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""llama.cpp's scale searches, applied to a quantized model as grid optimization
solvers (see brevitas.graph.grid_optimization).

  - .kernels   : the block searches;
  - .objective : what a fit is weighed by;
  - .solver    : the searches as solvers.

Importing this package registers the solvers in ``SCALE_SOLVER_REGISTRY``.
"""

from brevitas_examples.llm.gguf_export.search.objective import MAGNITUDE_FORMS
from brevitas_examples.llm.gguf_export.search.objective import SCOPES
from brevitas_examples.llm.gguf_export.search.solver import SCALE_PLACEMENTS

__all__ = ["MAGNITUDE_FORMS", "SCALE_PLACEMENTS", "SCOPES"]
