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

from functools import partial
from types import SimpleNamespace
from typing import Callable
from typing import Optional
from typing import Tuple
import warnings

import torch
import torch.nn as nn

from brevitas.graph.gpxq import accumulate_covariance
from brevitas.graph.gpxq import ModuleOptimizer
from brevitas.graph.grid_optimization.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.grid_optimization.utils import _build_grid
from brevitas.graph.grid_optimization.utils import _get_base_scale
from brevitas.graph.grid_optimization.utils import _grid_threshold
from brevitas.graph.grid_optimization.utils import _has_base_scale
from brevitas.graph.grid_optimization.utils import _has_zero_zero_point
from brevitas.graph.grid_optimization.utils import _is_float_weight_quant
from brevitas.graph.grid_optimization.utils import _is_int_weight_quant
from brevitas.graph.grid_optimization.utils import _set_base_scale
from brevitas.graph.grid_optimization.utils import compute_group_importance_permutation
from brevitas.graph.layerwise_hook import layerwise_hook_mode
from brevitas.graph.utils import is_quant_module
from brevitas.utils.torch_utils import StopFwdException

# How the solver uses the covariance matrix H (and G): 'dense' keeps the full matrices
# accumulated from calibration data, 'diagonal' keeps only their diagonal, and
# 'identity' ignores the data entirely (H=I, G=None, i.e. pure weight MSE).
COVARIANCE_MODES = ('dense', 'diagonal', 'identity')


class GridOptimizer:
    """Optimizes one layer's weight quantization grid, writing the result into its quantizer.

    optimize_layer takes the covariance matrices H (and optional G), runs the appropriate
    solver for the layer's granularity (per-channel or per-group) and configuration, and
    writes all scales at once. H and G are the covariance matrices of Eq. H,G-def
    (H = X_tilde^T X_tilde, G = X_tilde^T X), 2-D and in the layer's natural column
    order; G=None selects the self-activation objective (the solvers then use H in place
    of G). The solver family is selected by name from SCALE_SOLVER_REGISTRY (default
    'piso', see .solvers).

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
            covariance_mode: str = 'dense',
            solver: str = 'piso') -> None:
        GridOptimizer.validate_config(covariance_mode, solver)
        self.layer = layer
        self.solver_dtype = solver_dtype
        self.solver_device = solver_device
        self.tolerance = tolerance
        self.solver_batch_size = solver_batch_size
        self.per_group_greedy = per_group_greedy
        self.per_group_greedy_reorder = per_group_greedy_reorder
        self.covariance_mode = covariance_mode
        self._solver_family = SCALE_SOLVER_REGISTRY.get(solver)

    @staticmethod
    def validate_config(covariance_mode: str, solver: str) -> None:
        """Raise ValueError for an unknown covariance_mode or solver name.

        Drivers call it up front, so a bad configuration fails before calibration
        rather than when the first layer is built."""
        if covariance_mode not in COVARIANCE_MODES:
            raise ValueError(
                f"Unknown covariance_mode '{covariance_mode}'; expected one of {list(COVARIANCE_MODES)}."
            )
        # Raises ValueError listing the valid names on a miss.
        SCALE_SOLVER_REGISTRY.get(solver)

    @staticmethod
    def is_layer_valid(layer: nn.Module) -> bool:
        """Whether grid optimization supports layer: a signed, weight-quantized QuantLinear
        with a parameter-from-stats scale, per-channel or per-group, int (with zero
        zero-point) or float. Returns False (warning that the layer is skipped)
        instead of raising on any mismatch."""
        try:
            if not is_quant_module(layer):
                warnings.warn("Skipping grid optimization: layer is not a quant module.")
                return False
            if not layer.weight_quant.is_quant_enabled:
                warnings.warn(
                    "Skipping grid optimization: layer weight quantization is not enabled.")
                return False
            if not isinstance(layer, nn.Linear):
                warnings.warn("Skipping grid optimization: only nn.Linear layers are supported.")
                return False
            if not _has_base_scale(layer):
                warnings.warn(
                    "Skipping grid optimization: layer scale is not a "
                    "ParameterFromStatsFromParameterScaling.")
                return False
            if not layer.weight_quant.is_signed:
                warnings.warn("Skipping grid optimization: a signed weight quantizer is required.")
                return False
            if _is_float_weight_quant(layer.weight_quant):
                pass  # float weight quantizers have no zero-point requirement
            elif _is_int_weight_quant(layer.weight_quant):
                if not _has_zero_zero_point(layer):
                    warnings.warn(
                        "Skipping grid optimization: int weight quantizer must have a zero "
                        "zero-point.")
                    return False
            else:
                warnings.warn(
                    "Skipping grid optimization: only int and float weight quantizers are "
                    "supported.")
                return False
        except Exception:
            warnings.warn("Skipping grid optimization: layer validity check raised.")
            return False
        return True

    def cast_to_solver_dtype_device(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Move a tensor to the solver dtype/device, passing None through."""
        return None if t is None else t.to(self.solver_dtype).to(self.solver_device)

    def optimize_layer(self, H: torch.Tensor, G: Optional[torch.Tensor] = None) -> None:
        """Compute and write the optimized scales of the layer in one shot.

        Builds the quantization grid from the layer's quantizer, selects the
        matching solver of the family (per-channel vs per-group, full vs
        diagonal H, greedy sequential), runs it in batches over output
        channels, and stores optimal_s * threshold into the weight quantizer's
        scaling parameter. The layer params must be allocated by the caller.

        Solver-specific work is split across _select_solver (choose the function),
        _prepare_matrices (shape/permute H/G/W) and _bind_solver_matrices (bind
        its arguments); the rest is solver-agnostic.
        """
        layer = self.layer
        # Extract the quantization configuration from the layer's weight quantizer
        weight_quant = layer.weight_quant
        signed_scale = not weight_quant.restrict_scale_positive
        is_group = weight_quant.is_groupwise
        group_size = weight_quant.group_size if is_group else 0
        use_diag = self.covariance_mode != 'dense'
        is_greedy = is_group and self.per_group_greedy and not use_diag
        # Get layer weight shapes
        n_channels, channel_size = layer.weight.shape
        # Set batch size for the algorithm (number of channels whose scale is computed at the same time)
        batch_size = n_channels if self.solver_batch_size is None else self.solver_batch_size

        if is_group and channel_size % group_size != 0:
            raise ValueError(
                f"Channel size {channel_size} is not divisible by group_size {group_size}")

        self._warn_ignored_options(is_group, use_diag)

        # Prepare the matrices that the solver receives as input
        H = self.cast_to_solver_dtype_device(H)
        G = self.cast_to_solver_dtype_device(G)
        W = self.cast_to_solver_dtype_device(layer.weight.clone())

        if self.covariance_mode == 'identity':
            H = torch.ones(
                layer.weight.shape[1], device=self.solver_device,
                dtype=self.solver_dtype).unsqueeze(0)
            G = None
        elif self.covariance_mode == 'diagonal':
            H = H.squeeze().diag()
            if G is not None:
                G = G.squeeze().diag()

        # Generate unscaled grid in intX or float_eXmY
        grid = _build_grid(layer, dtype=H.dtype, device=H.device)

        # Select the solver and the arguments it expects (H/G dense, diagonal, or greedy)
        solver_fn, arg_kind = self._select_solver(is_group, use_diag, is_greedy)

        # Prepare the input matrices for the solver
        H, G, W, _, inv_block_perm = self._prepare_matrices(
            H, G, W, is_group, is_greedy, n_channels, channel_size, group_size)

        # Bind the solver and its arguments into a callable that takes (W, allow_negative_s)
        compute_scale_method = self._bind_solver_matrices(solver_fn, arg_kind, grid, H, G)

        optimal_s = torch.zeros_like(_get_base_scale(layer), dtype=H.dtype)

        with torch.inference_mode():
            for i in range(0, W.shape[0], batch_size):
                end = i + batch_size
                optimal_s[i:end] = compute_scale_method(
                    w_b=W[i:end], allow_negative_s=(signed_scale and (grid[-1] != -grid[0])))

        if inv_block_perm is not None:
            optimal_s = optimal_s[:, inv_block_perm]

        # Update the s on the weight quantizer
        _set_base_scale(layer, optimal_s * _grid_threshold(grid, weight_quant))

    def _warn_ignored_options(self, is_group: bool, use_diag: bool) -> None:
        """Warn when per_group_greedy is set but ignored (diagonal H, or
        per-channel quantization)."""
        if not self.per_group_greedy:
            return
        if is_group and use_diag:
            warnings.warn(
                f'per_group_greedy is ignored with covariance_mode={self.covariance_mode!r}: '
                'a diagonal H has no cross-group interactions to model.')
        elif not is_group:
            warnings.warn(
                'per_group_greedy equals True. However, per-channel quantization is being used. So it will be ignored.'
            )

    def _apply_group_greedy_permutation(
        self,
        H: torch.Tensor,
        G: Optional[torch.Tensor],
        W: torch.Tensor,
        group_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Permute H/G/W by group importance for the sequential greedy sweep.

        Groups are ordered by descending diag(H) per block (per_group_greedy_reorder)
        or left in natural order. Returns (H, G, W, perm, inv_block_perm); perm is
        the applied column order and inv_block_perm unshuffles the per-group scales.
        """
        if self.per_group_greedy_reorder:
            # permute H, G and weights to account for group importance
            # groups are processed in descending order of sum(diag(H)) per block
            perm, block_perm = compute_group_importance_permutation(H.squeeze().diag(), group_size)
        else:
            # natural / original ordering: groups processed 0..n_groups-1
            dim = H.squeeze().shape[-1]
            perm = torch.arange(dim, device=H.device)
            block_perm = torch.arange(dim // group_size, device=H.device)

        inv_block_perm = torch.empty_like(block_perm)
        inv_block_perm[block_perm] = torch.arange(len(block_perm), device=block_perm.device)

        H = H.squeeze()[:, perm]
        H = H[perm, :]
        if G is not None:
            G = G.squeeze()[:, perm]
            G = G[perm, :]
        W = W[:, perm]

        return H, G, W, perm, inv_block_perm

    def _select_solver(self, is_group: bool, use_diag: bool, is_greedy: bool):
        """Return (solver_fn, arg_kind) from the solver family for the config flags."""
        return self._solver_family.get_layer_solver(is_group, use_diag, is_greedy)

    def _prepare_matrices(
        self,
        H: torch.Tensor,
        G: Optional[torch.Tensor],
        W: torch.Tensor,
        is_group: bool,
        is_greedy: bool,
        n_channels: int,
        channel_size: int,
        group_size: int,
    ) -> Tuple[torch.Tensor,
               Optional[torch.Tensor],
               torch.Tensor,
               Optional[torch.Tensor],
               Optional[torch.Tensor]]:
        """Shape H/G/W for the selected solver.

        Greedy permutes H/G/W and returns (perm, inv_block_perm); group reshapes W
        into (n_channels, n_groups, group_size); per-channel leaves W as is.
        Non-greedy paths return perm = inv_block_perm = None.
        """
        if is_greedy:
            H, G, W, perm, inv_block_perm = self._apply_group_greedy_permutation(
                H, G, W, group_size)
            W = W.reshape(n_channels, channel_size // group_size, group_size)
            return H, G, W, perm, inv_block_perm
        if is_group:
            W = W.reshape(n_channels, channel_size // group_size, group_size)
        return H, G, W, None, None

    def _bind_solver_matrices(self, solver_fn, arg_kind: str, grid, H, G) -> Callable:
        """Bind grid/tolerance and the H/G args into a (w_b, allow_negative_s)
        callable. arg_kind 'diag' passes H_diag/G_diag; 'dense'/'greedy' pass H/G.
        """

        def as_dense(M):
            return None if M is None else M.squeeze()

        def as_diag(M):
            return None if M is None else (M.squeeze() if M.dim() > 1 else M)

        bound = partial(solver_fn, unscaled_grid=grid, eps=self.tolerance)
        if arg_kind == 'diag':
            return partial(bound, H_diag=as_diag(H), G_diag=as_diag(G))
        return partial(bound, H=as_dense(H), G=as_dense(G))


class InterleavedGridOptimizer(GridOptimizer):
    """Grid optimizer interleaved with error correction, held by GPTQ / Qronos.

    GPxQ builds one per supported layer through its grid_optimizer_factory (see
    for_gpxq) and calls resolve_blocksize, act_order_permutation and the on_* methods
    from its extension hooks, with the layer params already allocated.

    Per-group layers are group-interleaved unless layer_interleaved is set: each group's
    scale is optimized when error correction reaches its first column, on weights that
    already carry the error diffused by the preceding groups. Otherwise every scale of
    the layer is optimized before error correction (layer-interleaved).
    """

    def __init__(self, layer: nn.Module, layer_interleaved: bool = False, **kwargs) -> None:
        super().__init__(layer, **kwargs)
        self.group_interleaved = layer.weight_quant.is_groupwise and not layer_interleaved
        # Group-interleaved state, set between on_error_correction_starts and
        # on_layer_finished (see _init_group_state).
        self._group = None

    @classmethod
    def for_gpxq(cls, layer: nn.Module, **kwargs) -> Optional['InterleavedGridOptimizer']:
        """Build the grid optimizer that GPTQ / Qronos hold for layer, or return None
        (warning that the layer is skipped) when grid optimization does not support it.

        Meant as the grid_optimizer_factory of GPxQ, with kwargs bound by partial."""
        if not cls.is_layer_valid(layer):
            return None
        return cls(layer, **kwargs)

    def resolve_blocksize(self, blocksize: int) -> int:
        """Error-correction block size: the group size for per-group layers, so that
        every block covers exactly one quantization group; blocksize otherwise."""
        weight_quant = self.layer.weight_quant
        if weight_quant.is_groupwise:
            return weight_quant.group_size
        return blocksize

    def act_order_permutation(self, covariance_group: torch.Tensor) -> torch.Tensor:
        """Act-order column ordering that keeps each quantization group contiguous.

        Group-interleaved optimization sets a group's scale when its first column is
        reached, so it needs the groups processed contiguously."""
        perm, _ = compute_group_importance_permutation(
            x=covariance_group.diag(), group_size=self.layer.weight_quant.group_size)
        return perm

    def on_error_correction_starts(
            self, H: torch.Tensor, G: Optional[torch.Tensor], perm: torch.Tensor) -> None:
        """Optimize the layer scales on the fixed grid (layer-interleaved), or prepare
        the group-wise updates (group-interleaved). perm is the column order error
        correction follows. The caller may modify H and G in place once this returns,
        so whatever is kept for the group-wise updates is a copy."""
        if self.group_interleaved:
            self._init_group_state(H, G, perm)
        else:
            self.optimize_layer(H, G)

    def on_column_reached(self, column_idx: int) -> None:
        if self.group_interleaved:
            self.optimize_group(column_idx)

    def on_layer_finished(self) -> None:
        self._group = None

    def _init_group_state(
            self, H: torch.Tensor, G: Optional[torch.Tensor], perm: torch.Tensor) -> None:
        """Store the per-group solver's invariant inputs, in its dtype/device.

        Only w_updated changes between groups (error correction rewrites the
        weights as it goes), so everything else is prepared once instead of on
        every group:
          - group-diag reads only the covariance diagonals, so a diagonal/identity H
            is never materialized as a dense matrix;
          - group-greedy reads H/G/W in full, so casting them once avoids
            re-transferring the whole covariance matrix for every group;
          - group-dense only ever reads its own block, so H/G are left where they
            are and sliced before casting, keeping the per-group transfer small.
        perm is the column order error correction follows, used to map a column to
        its group.
        """
        weight_quant = self.layer.weight_quant
        # Generate unscaled grid in intX or float_eXmY
        grid = self.cast_to_solver_dtype_device(
            _build_grid(self.layer, dtype=H.dtype, device=H.device))
        use_diag = self.covariance_mode != 'dense'
        solver_fn, strategy = self._solver_family.get_group_solver(use_diag, self.per_group_greedy)
        group = SimpleNamespace(
            perm=perm,
            grid=grid,
            group_size=weight_quant.group_size,
            allow_negative_s=not weight_quant.restrict_scale_positive and (grid[-1] != -grid[0]),
            threshold=_grid_threshold(grid, weight_quant),
            solver_fn=solver_fn,
            strategy=strategy,
            H=None,
            G=None,
            W=None,
            H_diag=None,
            G_diag=None)

        if strategy == 'group-diag':
            if self.covariance_mode == 'identity':
                group.H_diag = torch.ones(
                    H.shape[-1], dtype=self.solver_dtype, device=self.solver_device)
            else:
                group.H_diag = self.cast_to_solver_dtype_device(H.diag())
                group.G_diag = self.cast_to_solver_dtype_device(G.diag()) if G is not None else None
        elif strategy == 'group-greedy':
            # copy=True: error correction may modify H and G in place
            group.H = H.to(dtype=self.solver_dtype, device=self.solver_device, copy=True)
            group.G = None if G is None else G.to(
                dtype=self.solver_dtype, device=self.solver_device, copy=True)
            # clone() is required because error correction modifies layer.weight in-place
            W = self.layer.weight_orig if hasattr(
                self.layer, 'weight_orig') else self.layer.weight.detach().clone()
            group.W = self.cast_to_solver_dtype_device(W)
        else:
            # clone(): error correction may modify H and G in place
            group.H = H.clone()
            group.G = None if G is None else G.clone()
        self._group = group

    def optimize_group(self, column_idx: int) -> None:
        """Optimize and write the scale of the group that owns column column_idx.

        Called for every column; only the first column of a group does work.
        column_idx is an index into the column order error correction follows; it is
        mapped back through perm to the group it belongs to. Picks the matching sweep
        (diagonal, sequential-greedy, or plain per-group) and writes that group's
        scale into the quantizer.
        """
        group = self._group
        if column_idx % group.group_size != 0:
            return
        group_idx = group.perm[column_idx] // group.group_size
        start = group_idx * group.group_size
        end = start + group.group_size

        optimal_s = self._bind_group_solver(start, end, group_idx)()

        # set the optimal s of this group in the QuantLayer
        _set_base_scale(self.layer, optimal_s * group.threshold, index=(slice(None), group_idx))

    def _bind_group_solver(self, start: int, end: int, group_idx: int) -> Callable:
        """Bind this group's arguments into a callable taking no further input.

        The counterpart of _bind_solver_matrices for optimize_group:
        _init_group_state already holds everything invariant, so the only
        tensor read here is the current (error-corrected) weight.
        """
        group = self._group
        bound = partial(
            group.solver_fn,
            unscaled_grid=group.grid,
            eps=self.tolerance,
            allow_negative_s=group.allow_negative_s)

        if group.strategy == 'group-diag':
            # H is diagonal → no cross-group correlations → use fast O(DL) algorithm
            return partial(
                bound,
                w_b=self.cast_to_solver_dtype_device(self.layer.weight[:, start:end]),
                H_diag=group.H_diag[start:end],
                G_diag=None if group.G_diag is None else group.G_diag[start:end])
        if group.strategy == 'group-greedy':
            # optimize single group taking into account previous groups
            return partial(
                bound,
                w_orig=group.W,
                w_updated=self.cast_to_solver_dtype_device(self.layer.weight),
                H=group.H,
                group_size=group.group_size,
                group_idx=group_idx,
                G=group.G)
        # optimize just the current group; slicing before the cast keeps the
        # per-group transfer to one block instead of the whole covariance matrix
        return partial(
            bound,
            w_b=self.cast_to_solver_dtype_device(self.layer.weight[:, start:end]),
            H=self.cast_to_solver_dtype_device(group.H[start:end, start:end]),
            G=None if group.G is None else self.cast_to_solver_dtype_device(
                group.G[start:end, start:end]))


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
            covariance_mode: str = 'dense',
            solver: str = 'piso') -> None:

        # Validate attributes received as arguments before touching the model
        GridOptimizer.validate_config(covariance_mode, solver)
        self._validate_args(covariance_mode)

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
            'solver': solver,}

        # NOTE: layerwise_hook_mode inherits from quantization_status_manager. As in
        # gpfq_mode, leaving activation and bias quantization enabled makes the
        # inherited manager a no-op, so the caller owns the quantization state.
        super().__init__(
            model=model,
            disable_act_quant=False,
            disable_bias_quant=False,
        )

    def _is_module_supported(self, module) -> bool:
        return GridOptimizer.is_layer_valid(module)

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
        # TODO (jpga) Move this to LLM args maybe? unsure about it, as the class could be used in other wrokflows
        if covariance_mode == 'diagonal':
            warnings.warn(
                'covariance_mode="diagonal": only the diagonal of H (and G) is used, so off-diagonal '
                'cross-weight interactions are discarded.')
        elif covariance_mode == 'identity':
            warnings.warn(
                'covariance_mode="identity": grid optimization is data-free (H=I, G=None, pure weight '
                'MSE). The calibration forward passes still run, but the computed H and G are ignored.'
            )
