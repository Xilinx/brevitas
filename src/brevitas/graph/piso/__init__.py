# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""PiSO: Piecewise Scale Optimization for weight quantization.

PiSO computes the weight quantization scale that minimizes the layer output
reconstruction error ||X w - s X_tilde q(w; s)||^2 exactly, under
round-to-nearest quantization. The grid assignment q(w; s) is piecewise
constant in the scale s, so the objective is quadratic on each interval
between consecutive "transition scales". PiSO sweeps these intervals, evaluates
the closed-form minimizer on each, and keeps the global best.

The package is split into:
  - .utils   : grid builders and the group-aware ordering helper;
  - .solvers : the algorithm core (the _piso_sweep_* interval solvers);
  - .core    : machinery to apply PiSO to a model (ScaleOptimizer /
               ScaleOptimizerGroupInterleaved, PiSOLayerHandler, optimize_scale_mode)
               and the GPTQ / Qronos integration mixins.

Entry points for the LLM example live in brevitas_examples.llm.llm_quant.piso.

Reference: "Optimal Post-Training Quantization Scales and Where to Find Them".
"""

from brevitas.graph.piso.core import GroupAwarePermGPTQ
from brevitas.graph.piso.core import GroupAwarePermQronos
from brevitas.graph.piso.core import GroupAwarePermutationMixin
from brevitas.graph.piso.core import optimize_scale_mode
from brevitas.graph.piso.core import PiSOGroupInterleavedGPTQ
from brevitas.graph.piso.core import PiSOGroupInterleavedQronos
from brevitas.graph.piso.core import PiSOLayerHandler
from brevitas.graph.piso.core import PiSOLayerInterleavedGPTQ
from brevitas.graph.piso.core import PiSOLayerInterleavedQronos
from brevitas.graph.piso.core import PiSOMixin
from brevitas.graph.piso.core import ScaleOptimizer
from brevitas.graph.piso.core import ScaleOptimizerGroupInterleaved
from brevitas.graph.piso.solvers import ChunkedPiSOSolverFamily
from brevitas.graph.piso.solvers import PiSOSolverFamily
from brevitas.graph.piso.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.piso.solvers import ScaleSolverFamily
