# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Applying grid optimization to a model.

GridOptimizer runs the solvers for one layer and writes the scales into its weight
quantizer. Its two subclasses decide where the covariance matrices come from and when
the scales are optimized:
  - StandaloneGridOptimizer is the hook that grid_optimization_mode attaches to each
    layer, like GPTQ with gptq_mode: it accumulates the covariance matrices from
    calibration data, then optimizes the layer's scales;
  - InterleavedGridOptimizer is held by GPTQ / Qronos, which hand it their covariance
    matrices during error correction (see the grid_optimizer_factory argument of
    brevitas.graph.gpxq.GPxQ).

Grid optimization chooses the weight quantization grid that minimizes the layer output
reconstruction error ||X w - s X_tilde q(w; s)||^2; currently, the scale s of the grid.
Two objectives are supported through the choice of covariance matrices
H = X_tilde^T X_tilde and G = X_tilde^T X:
  - self-activation (GPTQ-style): pass only H (G defaults to H);
  - cross-activation (Qronos/GPTAQ-style): pass both H and G.
Which activations X_tilde and X are (e.g. whether the previous layers are quantized)
is decided by whoever produces H and G: GPTQ / Qronos, or the caller of
grid_optimization_mode, which runs the forward passes.

The scale solvers live in .solvers and the grid builders / ordering helpers in
.utils. Entry points for the LLM example live in
brevitas_examples.llm.llm_quant.calibrate.
"""

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
import warnings

import torch
import torch.nn as nn

from brevitas.graph.gpxq import accumulate_covariance
from brevitas.graph.gpxq import ModuleOptimizer
from brevitas.graph.grid_optimization.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.grid_optimization.solvers import ScaleSolution
from brevitas.graph.grid_optimization.utils import _build_grid
from brevitas.graph.grid_optimization.utils import _get_scale_quant_grid
from brevitas.graph.grid_optimization.utils import _get_zero_point_quant_grid
from brevitas.graph.grid_optimization.utils import _grid_threshold
from brevitas.graph.grid_optimization.utils import _has_base_scale
from brevitas.graph.grid_optimization.utils import _has_zero_zero_point
from brevitas.graph.grid_optimization.utils import _set_base_scale
from brevitas.graph.grid_optimization.utils import _set_base_zero_point
from brevitas.graph.grid_optimization.utils import _set_nested_scale
from brevitas.graph.grid_optimization.utils import _set_nested_zero_point
from brevitas.graph.grid_optimization.utils import compute_group_importance_permutation
from brevitas.graph.layerwise_hook import layerwise_hook_mode
from brevitas.graph.utils import is_quant_module
from brevitas.utils.torch_utils import StopFwdException

# How the covariance matrix H (and G) is used: 'dense' keeps the full matrices
# accumulated from calibration data, for a solver that supports them, 'diagonal'
# keeps only their diagonal, and 'identity' ignores the data entirely (H=I, G=None,
# i.e. pure weight MSE).
COVARIANCE_MODES = ('dense', 'diagonal', 'identity')


def _restrict(M: Optional[torch.Tensor], columns: slice) -> Optional[torch.Tensor]:
    """A covariance matrix, or its diagonal, restricted to columns."""
    if M is None:
        return None
    return M[columns] if M.dim() == 1 else M[columns, columns]


def _quadratic_and_linear(
        W: torch.Tensor, H: torch.Tensor, G: Optional[torch.Tensor],
        covariance_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The quadratic term (H, or its diagonal) and the linear term b of every row of W.

    The objective ||X w - X_tilde r||^2 = r^T H r - 2 r^T G w + const, with
    H = X_tilde^T X_tilde and G = X_tilde^T X, or G = H for the self-activation
    objective. H and G may already be diagonals.
    """
    if covariance_mode == 'identity':
        return torch.ones(W.shape[1], dtype=W.dtype, device=W.device), W
    G = H if G is None else G
    if covariance_mode == 'diagonal':
        H = H if H.dim() == 1 else H.diagonal()
        return H, W * (G if G.dim() == 1 else G.diagonal())
    return H, W @ G.T


class GridOptimizer:
    """Optimizes one layer's weight quantization grid, writing the result into its quantizer.

    optimize_layer takes the covariance matrices H (and optional G), 2-D and in the
    layer's natural column order (H = X_tilde^T X_tilde, G = X_tilde^T X; G=None
    selects the self-activation objective). It turns them into the quadratic and the
    linear term of the objective every solver minimises (see ScaleSolverFamily),
    according to covariance_mode, and hands the solver the groups of the layer: a
    per-channel layer is a single group per row. The solver family is selected by
    name from SCALE_SOLVER_REGISTRY (default 'piso', see .solvers), and solver_kwargs
    are passed on to every call of its solver.

    With per_group_greedy, the groups are solved one unit at a time (a group, or the
    run of groups sharing a step on a nested grid, see utils.NestedGrid), in order
    of importance with per_group_greedy_reorder, each against the reconstruction of
    the units solved before it. That only differs from solving them all at once with
    a full covariance matrix.

    Built for a single layer, which must satisfy is_layer_valid. Its subclasses decide
    where the covariance matrices come from and when the scales are optimized:
    StandaloneGridOptimizer accumulates them from calibration data, and
    InterleavedGridOptimizer receives them from GPTQ / Qronos.
    """

    def __init__(
            self,
            layer: nn.Module,
            solver_dtype: torch.dtype = torch.float32,
            solver_device: str = 'cuda',
            tolerance: float = 1e-5,
            solver_batch_size: Optional[int] = None,
            per_group_greedy: bool = False,
            per_group_greedy_reorder: bool = True,
            covariance_mode: str = 'diagonal',
            solver: str = 'piso',
            solver_kwargs: Optional[Dict[str, Any]] = None) -> None:
        GridOptimizer.validate_config(covariance_mode, solver)
        self.layer = layer
        self.solver_dtype = solver_dtype
        self.solver_device = solver_device
        self.tolerance = tolerance
        self.solver_batch_size = solver_batch_size
        self.per_group_greedy = per_group_greedy
        self.per_group_greedy_reorder = per_group_greedy_reorder
        self.covariance_mode = covariance_mode
        self.solver_kwargs = dict(solver_kwargs or {})
        self._solver_family = SCALE_SOLVER_REGISTRY.get(solver)

    @staticmethod
    def validate_config(covariance_mode: str, solver: str) -> None:
        """Raise ValueError for an unknown covariance_mode or solver name, or a solver
        that cannot use the covariance mode.

        Drivers call it up front, so a bad configuration fails before calibration
        rather than when the first layer is built."""
        if covariance_mode not in COVARIANCE_MODES:
            raise ValueError(
                f"Unknown covariance_mode '{covariance_mode}'; expected one of {list(COVARIANCE_MODES)}."
            )
        # Raises ValueError listing the valid names on a miss.
        family = SCALE_SOLVER_REGISTRY.get(solver)
        if covariance_mode == 'dense' and not family.supports_dense:
            raise ValueError(
                f"Solver '{solver}' only uses the diagonal of the covariance matrix; use "
                "covariance_mode 'diagonal' or 'identity'.")

    @staticmethod
    def is_layer_valid(layer: nn.Module, solver: str = 'piso') -> bool:
        """Whether grid optimization with solver supports layer: a weight-quantized
        QuantLinear with a parameter-from-stats scale, whose quantizer the solver family
        accepts (see ScaleSolverFamily.check_layer). Returns False (warning that the
        layer is skipped) instead of raising on any mismatch."""
        if not is_quant_module(layer):
            warnings.warn("Skipping grid optimization: layer is not a quant module.")
            return False
        if not layer.weight_quant.is_quant_enabled:
            warnings.warn("Skipping grid optimization: layer weight quantization is not enabled.")
            return False
        if not isinstance(layer, nn.Linear):
            warnings.warn("Skipping grid optimization: only nn.Linear layers are supported.")
            return False
        if not _has_base_scale(layer):
            warnings.warn(
                "Skipping grid optimization: layer scale is not a "
                "ParameterFromStatsFromParameterScaling.")
            return False
        reason = SCALE_SOLVER_REGISTRY.get(solver).check_layer(layer)
        if reason is not None:
            warnings.warn(f"Skipping grid optimization: {reason}.")
            return False
        return True

    def cast_to_solver_dtype_device(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Move a tensor to the solver dtype/device, passing None through."""
        return None if t is None else t.to(self.solver_dtype).to(self.solver_device)

    def _unit_columns(self) -> int:
        """Columns of the smallest set of groups whose scales can be chosen on their own:
        a group, or the run of groups sharing a step on a nested grid. A per-channel
        layer is a single group spanning the row."""
        weight_quant = self.layer.weight_quant
        if not weight_quant.is_groupwise:
            return self.layer.weight.shape[1]
        nested = _get_scale_quant_grid(self.layer)
        return weight_quant.group_size * (1 if nested is None else nested.groups_per_step)

    def optimize_layer(self, H: torch.Tensor, G: Optional[torch.Tensor] = None) -> None:
        """Compute and write the optimized scales of the layer.

        The layer params must be allocated by the caller.
        """
        W = self.cast_to_solver_dtype_device(self.layer.weight.detach())
        H, b = _quadratic_and_linear(
            W,
            self.cast_to_solver_dtype_device(H),
            self.cast_to_solver_dtype_device(G),
            self.covariance_mode)
        if not self.per_group_greedy:
            self._optimize_columns(W, b, H, slice(0, W.shape[1]))
            return
        unit = self._unit_columns()
        order = torch.arange(W.shape[1] // unit)
        if self.per_group_greedy_reorder:
            H_diag = H if H.dim() == 1 else H.diagonal()
            _, order = compute_group_importance_permutation(H_diag, unit)
        for u in order.tolist():
            columns = slice(u * unit, (u + 1) * unit)
            self._optimize_columns(W[:, columns], b[:, columns], _restrict(H, columns), columns)
            if H.dim() == 2:
                # The units still to be solved see this one through their cross terms
                # with it, 2 r_unit^T H[unit, rest] r_rest, which move into their b.
                r = self.cast_to_solver_dtype_device(self.layer.quant_weight().value[:, columns])
                b -= r @ H[columns]

    def _optimize_columns(
            self, w: torch.Tensor, b: torch.Tensor, H: torch.Tensor, columns: slice) -> None:
        """Solve and write the scales of the groups spanning the layer's columns.

        w, b and H are already restricted to those columns.
        """
        layer = self.layer
        weight_quant = layer.weight_quant
        n_rows, n_columns = w.shape
        group_size = weight_quant.group_size if weight_quant.is_groupwise else n_columns
        w = w.reshape(n_rows, -1, group_size)
        b = b.reshape(w.shape)
        grid = _build_grid(layer, dtype=w.dtype, device=w.device)
        allow_negative_s = not weight_quant.restrict_scale_positive and bool(grid[-1] != -grid[0])
        with_offset = not _has_zero_zero_point(layer)
        # Only a layer whose scales are quantized on a nested grid passes its grids, and
        # only a family whose check_layer accepts such a layer receives them.
        nested = {}
        scale_quant_grid = _get_scale_quant_grid(layer)
        if scale_quant_grid is not None:
            nested = dict(
                scale_quant_grid=scale_quant_grid,
                zero_point_quant_grid=_get_zero_point_quant_grid(layer))
        batch_size = n_rows if self.solver_batch_size is None else self.solver_batch_size
        solutions = []
        with torch.inference_mode():
            for i in range(0, n_rows, batch_size):
                rows = slice(i, i + batch_size)
                solution = self._solver_family.solve(
                    w[rows],
                    b[rows],
                    H,
                    grid,
                    allow_negative_s=allow_negative_s,
                    with_offset=with_offset,
                    eps=self.tolerance,
                    **nested,
                    **self.solver_kwargs)
                if isinstance(solution, torch.Tensor):
                    solution = ScaleSolution(scale=solution)
                solutions.append(solution)

        # Concatenated outside inference mode, so that what is written into the layer
        # is an ordinary tensor.
        def concat(field):
            parts = [getattr(s, field) for s in solutions]
            return None if parts[0] is None else torch.cat(parts)

        groups = slice(columns.start // group_size, columns.stop // group_size)
        threshold = _grid_threshold(grid, weight_quant)
        scale, scale_step = concat('scale'), concat('scale_step')
        offset, offset_step = concat('offset'), concat('offset_step')
        if scale_step is None:
            _set_base_scale(layer, scale * threshold, groups)
        else:
            _set_nested_scale(layer, scale * threshold, scale_step * threshold, groups)
        if offset is not None:
            if offset_step is None:
                _set_base_zero_point(layer, offset, groups)
            else:
                _set_nested_zero_point(layer, offset, offset_step, groups)


class InterleavedGridOptimizer(GridOptimizer):
    """Grid optimizer interleaved with error correction, held by GPTQ / Qronos.

    GPxQ builds one per supported layer through its grid_optimizer_factory (see
    for_gpxq) and calls resolve_blocksize, act_order_permutation and the on_* methods
    from its extension hooks, with the layer params already allocated.

    Per-group layers are group-interleaved unless layer_interleaved is set: the scales
    of each unit (a group, or the run of groups sharing a step on a nested grid) are
    optimized when error correction reaches its first column, on weights that already
    carry the error diffused by the preceding units, against the unit's own block of
    the covariance matrices. Otherwise every scale of the layer is optimized before
    error correction (layer-interleaved).
    """

    def __init__(self, layer: nn.Module, layer_interleaved: bool = False, **kwargs) -> None:
        super().__init__(layer, **kwargs)
        self.group_interleaved = layer.weight_quant.is_groupwise and not layer_interleaved
        if self.group_interleaved and self.per_group_greedy:
            raise ValueError(
                "per_group_greedy chooses the order the groups are solved in, which "
                "group-interleaved optimization takes from error correction; set "
                "layer_interleaved to combine them.")
        # The covariance matrices (or their diagonals) and the column order of error
        # correction, kept between on_error_correction_starts and on_layer_finished.
        self._state = None

    @classmethod
    def for_gpxq(cls,
                 layer: nn.Module,
                 solver: str = 'piso',
                 **kwargs) -> Optional['InterleavedGridOptimizer']:
        """Build the grid optimizer that GPTQ / Qronos hold for layer, or return None
        (warning that the layer is skipped) when grid optimization does not support it.

        Meant as the grid_optimizer_factory of GPxQ, with kwargs bound by partial."""
        if not cls.is_layer_valid(layer, solver):
            return None
        return cls(layer, solver=solver, **kwargs)

    def resolve_blocksize(self, blocksize: int) -> int:
        """Error-correction block size: one unit for per-group layers, so that every
        block covers exactly the groups whose scales are optimized together;
        blocksize otherwise."""
        if self.layer.weight_quant.is_groupwise:
            return self._unit_columns()
        return blocksize

    def act_order_permutation(self, covariance_group: torch.Tensor) -> torch.Tensor:
        """Act-order column ordering that keeps each unit contiguous.

        Group-interleaved optimization sets a unit's scales when its first column is
        reached, so it needs the units processed contiguously."""
        perm, _ = compute_group_importance_permutation(
            x=covariance_group.diag(), group_size=self._unit_columns())
        return perm

    def on_error_correction_starts(
            self, H: torch.Tensor, G: Optional[torch.Tensor], perm: torch.Tensor) -> None:
        """Optimize the layer scales (layer-interleaved), or keep what the unit-wise
        updates need (group-interleaved). perm is the column order error correction
        follows. The caller may modify H and G in place once this returns, so what is
        kept is a copy, of the diagonals only when that is all the solver reads."""
        if not self.group_interleaved:
            self.optimize_layer(H, G)
            return
        if self.covariance_mode != 'dense':
            H = H.diagonal()
            G = None if G is None else G.diagonal()
        self._state = (H.clone(), None if G is None else G.clone(), perm)

    def on_column_reached(self, column_idx: int) -> None:
        """Optimize the scales of the unit whose first column column_idx is, an index
        into the column order error correction follows."""
        if not self.group_interleaved:
            return
        unit = self._unit_columns()
        if column_idx % unit != 0:
            return
        H, G, perm = self._state
        start = int(perm[column_idx]) // unit * unit
        columns = slice(start, start + unit)
        w = self.cast_to_solver_dtype_device(self.layer.weight.detach()[:, columns])
        H, b = _quadratic_and_linear(
            w,
            self.cast_to_solver_dtype_device(_restrict(H, columns)),
            self.cast_to_solver_dtype_device(_restrict(G, columns)),
            self.covariance_mode)
        self._optimize_columns(w, b, H, columns)

    def on_layer_finished(self) -> None:
        self._state = None


class StandaloneGridOptimizer(GridOptimizer, ModuleOptimizer):
    """Standalone grid optimizer, the per-layer hook of grid_optimization_mode.

    update_batch accumulates the covariance matrices H and G from the layer inputs, and
    single_layer_update then optimizes every scale of the layer at once.
    cross_act_objective selects whether G is accumulated (see grid_optimization_mode
    for the forward passes it expects), and matrix_dtype is the dtype of H and G.
    """

    def __init__(
            self,
            layer: nn.Module,
            name: str,
            len_parallel_layers: int = 1,
            cross_act_objective: bool = False,
            matrix_dtype: torch.dtype = torch.float32,
            **kwargs) -> None:
        GridOptimizer.__init__(self, layer, **kwargs)
        # The covariance matrices are kept on the device of the layer weights, and the scales are
        # optimized without a copy of the original weights.
        ModuleOptimizer.__init__(
            self,
            layer,
            name,
            len_parallel_layers,
            create_weight_orig=False,
            device='same',
            dtype=matrix_dtype)
        self.cross_act_objective = cross_act_objective
        # Allocated by the first update_batch call, so that only the layers being
        # calibrated hold them.
        self.H = None
        self.G = None
        self.nsamples = 0
        self.nsamples_G = 0
        self.quant_input = None

    def update_batch(self, module, input, current_layer):
        current_layer.layer_names.add(self.name)
        inp_processed = self.process_input(input)
        if self.H is None:
            self.H = torch.zeros((self.groups, self.columns, self.columns),
                                 device=self.device,
                                 dtype=self.dtype)
            self.G = torch.zeros_like(self.H) if self.cross_act_objective else None
        if self.cross_act_objective and not module.weight_quant.is_quant_enabled:
            # Float pass: accumulate G = X_tilde^T X, pairing this input with the
            # quantized input stored by the previous pass.
            self.nsamples_G = accumulate_covariance(
                self.G, self.nsamples_G, self.quant_input, inp_processed)
            self.quant_input = None
        else:
            # Accumulate H = X_tilde^T X_tilde. For the cross-activation objective, also
            # store this input to pair it with the float input of the next pass.
            if self.cross_act_objective:
                if self.quant_input is not None:
                    raise RuntimeError(
                        "Expected quant_input to be consumed by the float pass before the "
                        "next quantized pass; got a stale quant_input.")
                self.quant_input = inp_processed.to(dtype=self.H.dtype, copy=True)
            self.nsamples = accumulate_covariance(self.H, self.nsamples, inp_processed)
        raise StopFwdException

    def single_layer_update(self):
        # Allocate params if offloaded (e.g., meta device with accelerate)
        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        # The layer is an nn.Linear (see is_layer_valid), so there is a single group.
        G = None if self.G is None else self.G[0]
        self.optimize_layer(self.H[0], G)
        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)
        self.H = self.G = None


class grid_optimization_mode(layerwise_hook_mode):
    """Context manager that runs standalone (decoupled) grid optimization over a model.

    On entry it selects the supported layers, attaches a StandaloneGridOptimizer hook to
    each, and swaps in a forward that catches the early-stop exception. The caller
    feeds calibration data through the model to accumulate covariance matrices, then calls update
    to optimize every layer's scales. This is the standalone path; the interleaved path
    instead runs inside GPTQ/Qronos (see InterleavedGridOptimizer).

    Like gpfq_mode, it never enables or disables quantization: the caller runs
    every forward pass under the quantization state its objective needs.
      - cross_act_objective=False: one pass per batch. Whether the layers before
        the current one are quantized (self-activation) or left float (unquantized)
        is up to the caller.
      - cross_act_objective=True: per batch, a quantized pass first (X_tilde, whose
        input the hook stores), then a float pass with all quantization disabled
        (X, which accumulates G).
    update has to run outside that disabling: a layer's weight quantization must be
    enabled for its scales to be written.

    It keeps the inherited solves_mismatched_objective = False on purpose: block-wise
    drivers then cache a single stream of block inputs, and the caller runs both passes
    of the cross-activation objective on it.
    """

    def __init__(
            self,
            model,
            solver_dtype: torch.dtype = torch.float32,
            matrix_dtype: torch.dtype = torch.float32,
            tolerance: float = 1e-5,
            solver_batch_size: Optional[int] = None,
            cross_act_objective: bool = False,
            per_group_greedy: bool = False,
            per_group_greedy_reorder: bool = True,
            covariance_mode: str = 'diagonal',
            solver: str = 'piso',
            solver_kwargs: Optional[Dict[str, Any]] = None) -> None:

        # Validate attributes received as arguments before touching the model
        GridOptimizer.validate_config(covariance_mode, solver)
        self._validate_args(covariance_mode)

        self.solver = solver
        self.grid_optimizer_kwargs = {
            'cross_act_objective': cross_act_objective,
            'matrix_dtype': matrix_dtype,
            'solver_dtype': solver_dtype,
            'solver_device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'tolerance': tolerance,
            'solver_batch_size': solver_batch_size,
            'per_group_greedy': per_group_greedy,
            'per_group_greedy_reorder': per_group_greedy_reorder,
            'covariance_mode': covariance_mode,
            'solver': solver,
            'solver_kwargs': solver_kwargs,}

        # NOTE: layerwise_hook_mode inherits from quantization_status_manager. As in
        # gpfq_mode, leaving activation and bias quantization enabled makes the
        # inherited manager a no-op, so the caller owns the quantization state.
        super().__init__(
            model=model,
            disable_act_quant=False,
            disable_bias_quant=False,
        )

    def _is_module_supported(self, module) -> bool:
        return GridOptimizer.is_layer_valid(module, self.solver)

    def catch_stopfwd(self, *args, **kwargs):
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass

    @torch.no_grad()
    def __enter__(self):
        return super().__enter__()

    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        # create_weight_orig is part of the shared initialize_module_optimizer contract
        # but unused: grid optimization keeps no copy of the original weights.
        return StandaloneGridOptimizer(
            layer, name, len_parallel_layers=len_parallel_layers, **self.grid_optimizer_kwargs)

    def _validate_args(self, covariance_mode: str) -> None:
        if covariance_mode == 'identity':
            warnings.warn(
                'covariance_mode="identity": grid optimization is data-free (H=I, G=None, pure weight '
                'MSE). The calibration forward passes still run, but the computed H and G are ignored.'
            )
