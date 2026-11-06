# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Grid optimization for weight quantization.

Grid optimization chooses the weight quantization grid that minimizes the layer
output reconstruction error ||X w - s X_tilde q(w; s)||^2 on calibration data.
Currently it optimizes the scale s. The scale solvers are selected by name from
SCALE_SOLVER_REGISTRY; the default is PiSO (Piecewise Scale Optimization). Under
round-to-nearest quantization the assignment q(w; s) is piecewise constant in s,
so the objective is quadratic on each interval between consecutive "transition
scales". PiSO sweeps these intervals, evaluates the closed-form minimizer on each,
and keeps the global best.

The package is split into:
  - .utils   : grid builders, scale accessors and the group-aware ordering helper;
  - .solvers : the scale solvers (PiSO and its chunked variant);
  - .core    : the per-layer GridOptimizer (the solver) and its two uses:
               StandaloneGridOptimizer, attached by grid_optimization_mode, and
               InterleavedGridOptimizer, held by GPTQ / Qronos to interleave grid
               optimization with error correction.

Entry points for the LLM example live in brevitas_examples.llm.llm_quant.calibrate.

PiSO reference: "Optimal Post-Training Quantization Scales and Where to Find Them"
(https://arxiv.org/abs/2606.10890).
"""

from brevitas.graph.grid_optimization.core import grid_optimization_mode
from brevitas.graph.grid_optimization.core import GridOptimizer
from brevitas.graph.grid_optimization.core import InterleavedGridOptimizer
from brevitas.graph.grid_optimization.core import StandaloneGridOptimizer
from brevitas.graph.grid_optimization.solvers import ChunkedPiSOSolverFamily
from brevitas.graph.grid_optimization.solvers import PiSOSolverFamily
from brevitas.graph.grid_optimization.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.grid_optimization.solvers import ScaleSolverFamily
