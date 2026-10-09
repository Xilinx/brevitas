# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Scale-optimization solvers for PiSO.

Each solver computes the reconstruction-optimal weight quantization scale by
sweeping the intervals on which the round-to-nearest assignment q(w; s) is
constant and evaluating the closed-form per-interval minimizer.

Layout:
  - .common       : shared interval-sweep helpers + the solver registry and the
                    ScaleSolverFamily base class;
  - .piso         : the _piso_sweep_* variants (paper Algorithm 1) and
                    PiSOSolverFamily, registered as 'piso';
  - .chunked_piso : the chunked reformulations and ChunkedPiSOSolverFamily,
                    registered as 'chunked-piso' (positive scales only).

Only the diagonal-H slots are implemented; the dense-H ones raise
NotImplementedError. Importing this package registers every family in
SCALE_SOLVER_REGISTRY, so ScaleOptimizer can resolve them by name.

Reference: "Optimal Post-Training Quantization Scales and Where to Find Them".
"""

from brevitas.graph.piso.solvers.chunked_piso import _chunked_piso_sweep_per_channel_diag
from brevitas.graph.piso.solvers.chunked_piso import _chunked_piso_sweep_per_group_diag
from brevitas.graph.piso.solvers.chunked_piso import ChunkedPiSOSolverFamily
from brevitas.graph.piso.solvers.common import SCALE_SOLVER_REGISTRY
from brevitas.graph.piso.solvers.common import ScaleSolverFamily
from brevitas.graph.piso.solvers.piso import _piso_sweep_per_channel_diag
from brevitas.graph.piso.solvers.piso import _piso_sweep_per_group_diag
from brevitas.graph.piso.solvers.piso import PiSOSolverFamily

__all__ = [
    "SCALE_SOLVER_REGISTRY",
    "ScaleSolverFamily",
    "PiSOSolverFamily",
    "ChunkedPiSOSolverFamily",
    "_piso_sweep_per_channel_diag",
    "_piso_sweep_per_group_diag",
    "_chunked_piso_sweep_per_channel_diag",
    "_chunked_piso_sweep_per_group_diag",]
