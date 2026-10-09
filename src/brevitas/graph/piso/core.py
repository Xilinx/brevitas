# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Applying PiSO to a model.

The layer-level and group-wise updaters (ScaleOptimizer / ScaleOptimizerGroupInterleaved)
that run the solvers and write scales into a layer's weight quantizer, the
calibration hook (PiSOLayerHandler) and standalone driver (optimize_scale_mode),
and the mixins that interleave PiSO with GPTQ / Qronos error correction.

PiSO computes the weight quantization scale that minimizes the layer output
reconstruction error ||X w - s X_tilde q(w; s)||^2 exactly, under
round-to-nearest quantization. Two objectives are supported through the choice of
Hessians H = X_tilde^T X_tilde and G = X_tilde^T X (see ScaleOptimizer):
  - self-activation (GPTQ-style): pass only H (G defaults to H);
  - cross-activation (Qronos/GPTAQ-style): pass both H and G.

The algorithm core (the _piso_sweep_* solvers) lives in .solvers and the grid
builders / ordering helpers in .utils. Entry points for the LLM example live in
brevitas_examples.llm.llm_quant.piso.

Reference: "Optimal Post-Training Quantization Scales and Where to Find Them".
"""

from functools import partial
import math
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
import warnings

import torch
import torch.nn as nn

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.calibrate import QuantizationStatusManager
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gpxq import process_layer_input
from brevitas.graph.layerwise_hook import layerwise_hook_mode
from brevitas.graph.piso.solvers import SCALE_SOLVER_REGISTRY
from brevitas.graph.piso.utils import _build_grid
from brevitas.graph.piso.utils import _get_base_scale
from brevitas.graph.piso.utils import _grid_threshold
from brevitas.graph.piso.utils import _has_base_scale
from brevitas.graph.piso.utils import _has_zero_zero_point
from brevitas.graph.piso.utils import _is_float_weight_quant
from brevitas.graph.piso.utils import _is_int_weight_quant
from brevitas.graph.piso.utils import _set_base_scale
from brevitas.graph.piso.utils import compute_group_importance_permutation
from brevitas.graph.qronos import Qronos
from brevitas.graph.utils import is_quant_module
from brevitas.utils.torch_utils import StopFwdException

# How the solver uses the Hessian H (and G): 'dense' keeps the full matrices
# accumulated from calibration data, 'diagonal' keeps only their diagonal, and
# 'identity' ignores the data entirely (H=I, G=None, i.e. pure weight MSE).
HESSIAN_MODES = ('dense', 'diagonal', 'identity')


class ScaleOptimizer:
    """Applies PiSO to a layer, writing optimized scales into its weight quantizer.

    This is the layer-level updater: single_layer_update takes the Hessians
    H (and optional G), runs the appropriate solver for the layer's granularity
    (per-channel or per-group) and configuration, and writes all scales at once.
    It therefore holds no per-group state. The solver family is selected by name
    from SCALE_SOLVER_REGISTRY (default 'piso').

    ScaleOptimizerGroupInterleaved is the alternative for interleaving with error
    correction: it defers work, updating one group's scale at a time as GPTQ/Qronos
    reaches it. The error-correction code (gptq.py / qronos.py) drives both through
    a common protocol (should_update_group, single_group_update, clear_scale_optimizer_params)
    so it never needs to know which concrete class it holds. The base class provides
    no-op versions of the group-wise hooks.
    """

    def __init__(
            self,
            solver_dtype: torch.dtype = torch.float32,
            solver_device: str = 'cuda',
            tolerance: float = 1e-5,
            solver_batch_size: Optional[int] = None,
            cross_act_objective: bool = False,
            per_group_greedy: bool = False,
            per_group_greedy_reorder: bool = True,
            hessian_mode: str = 'dense',
            solver: str = 'piso'):
        self.solver_dtype = solver_dtype
        self.solver_device = solver_device
        self.tolerance = tolerance
        self.solver_batch_size = solver_batch_size
        self.cross_act_objective = cross_act_objective
        self.per_group_greedy = per_group_greedy
        self.per_group_greedy_reorder = per_group_greedy_reorder
        self.hessian_mode = hessian_mode
        self.solver = solver
        if hessian_mode not in HESSIAN_MODES:
            raise ValueError(
                f"Unknown hessian_mode '{hessian_mode}'; expected one of {list(HESSIAN_MODES)}.")
        # Resolve the solver family (raises ValueError listing valid names on miss).
        self._solver_family = SCALE_SOLVER_REGISTRY.get(solver)

    def cast_to_solver_dtype_device(self, t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Move a tensor to the solver dtype/device, passing None through."""
        return None if t is None else t.to(self.solver_dtype).to(self.solver_device)

    def single_layer_update(
            self,
            layer: nn.Module,
            H: Optional[torch.Tensor] = None,
            G: Optional[torch.Tensor] = None,
            skip_offload: bool = False):
        """Compute and write the optimized scales for layer in one shot.

        Builds the quantization grid from the layer's quantizer, selects the
        matching _piso_sweep_* solver (per-channel vs per-group, full vs
        diagonal H, greedy sequential), runs it in batches over output
        channels, and stores optimal_s * threshold into the weight quantizer's
        scaling parameter. H/G are the Hessians of Eq. H,G-def; if
        cross_act_objective is set, G is required. skip_offload means the caller
        manages the layer params: it has allocated them already and will offload
        them later, so neither is done here.

        Solver-specific work is split across _select_solver (choose the function),
        _prepare_matrices (shape/permute H/G/W) and _bind_solver_matrices (bind
        its arguments); the rest is solver-agnostic.
        """
        if not ScaleOptimizer.is_layer_valid(layer):
            return

        if self.cross_act_objective and G is None:
            raise ValueError("Expected matrix G to not be None for cross_act_objective=True")

        # Extract the quantization configuration from the layer's weight quantizer
        weight_quant = layer.weight_quant
        signed_scale = not weight_quant.restrict_scale_positive
        is_group = weight_quant.is_groupwise
        group_size = weight_quant.group_size if is_group else 0
        use_diag = self.hessian_mode != 'dense'
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

        if self.hessian_mode == 'identity':
            H = torch.ones(
                layer.weight.shape[1], device=self.solver_device,
                dtype=self.solver_dtype).unsqueeze(0)
            G = None
        elif self.hessian_mode == 'diagonal':
            if H is not None:
                H = H.squeeze().diag()
            else:
                H = torch.ones(
                    layer.weight.shape[1], device=self.solver_device, dtype=self.solver_dtype)
            if G is not None:
                G = G.squeeze().diag()

        # Generate unscaled grid in intX or float_eXmY
        grid = _build_grid(layer, dtype=H.dtype, device=H.device)

        # Allocate params if offloaded (e.g., meta device with accelerate). A caller
        # passing skip_offload owns the whole params lifecycle and has already
        # allocated (GPTQ/Qronos do so at the top of single_layer_update), so
        # allocating again would re-materialize the weights from accelerate's
        # weights_map and discard anything written since.
        if not skip_offload and hasattr(layer, 'allocate_params'):
            layer.allocate_params(layer)

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

        # Offload params back to free memory
        if not skip_offload and hasattr(layer, 'offload_params'):
            layer.offload_params(layer)

        # Clean-up of input matrices
        del H
        if G is not None:
            del G

    def _warn_ignored_options(self, is_group: bool, use_diag: bool) -> None:
        """Warn when per_group_greedy is set but ignored (diagonal H, or
        per-channel quantization)."""
        if not self.per_group_greedy:
            return
        if is_group and use_diag:
            warnings.warn(
                f'per_group_greedy is ignored with hessian_mode={self.hessian_mode!r}: '
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
        if self.cross_act_objective:
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

    @staticmethod
    def is_layer_valid(layer: nn.Module) -> bool:
        """Whether PiSO supports layer: a signed, weight-quantized QuantLinear
        with a parameter-from-stats scale, per-channel or per-group, int (with zero
        zero-point) or float. Returns False (warning that the layer is skipped)
        instead of raising on any mismatch."""
        try:
            if not is_quant_module(layer):
                warnings.warn("Skipping scale optimization: layer is not a quant module.")
                return False
            if not layer.weight_quant.is_quant_enabled:
                warnings.warn(
                    "Skipping scale optimization: layer weight quantization is not enabled.")
                return False
            if not isinstance(layer, nn.Linear):
                warnings.warn("Skipping scale optimization: PiSO only supports nn.Linear layers.")
                return False
            if not _has_base_scale(layer):
                warnings.warn(
                    "Skipping scale optimization: layer scale is not a "
                    "ParameterFromStatsFromParameterScaling.")
                return False
            if not layer.weight_quant.is_signed:
                warnings.warn(
                    "Skipping scale optimization: PiSO requires a signed weight quantizer.")
                return False
            if _is_float_weight_quant(layer.weight_quant):
                pass  # float weight quantizers have no zero-point requirement
            elif _is_int_weight_quant(layer.weight_quant):
                if not _has_zero_zero_point(layer):
                    warnings.warn(
                        "Skipping scale optimization: int weight quantizer must have a zero "
                        "zero-point.")
                    return False
            else:
                warnings.warn(
                    "Skipping scale optimization: only int and float weight quantizers are "
                    "supported.")
                return False
        except Exception:
            warnings.warn("Skipping scale optimization: layer validity check raised.")
            return False
        return True

    def should_update_group(self, column_idx: int) -> bool:
        """Whether error correction reaching column_idx should trigger a group
        scale update. Always False for the layer-level updater, which computes
        every scale in single_layer_update rather than per group."""
        return False

    def single_group_update(self, incoming_idx: int) -> None:
        """Optimize the scale of the group containing column incoming_idx.

        No-op here: the layer-level updater computes every scale in
        single_layer_update. Error-correction code may call this on any updater;
        only ScaleOptimizerGroupInterleaved acts on it.
        """
        pass

    def clear_scale_optimizer_params(self) -> None:
        """Release any per-layer state held for interleaved updates.

        No-op here (the layer-level updater keeps none). Overridden by
        ScaleOptimizerGroupInterleaved and safe to call unconditionally.
        """
        pass


class ScaleOptimizerGroupInterleaved(ScaleOptimizer):
    """Group-wise PiSO updater for interleaving with error correction.

    Instead of computing all scales at once, this defers to per-group updates:
    init_scale_optimizer_params stores the layer's Hessians/grid/permutation, then GPTQ or
    Qronos calls single_group_update as it reaches each group (so the scale is
    fit on weights already carrying earlier groups' error-diffusion updates), and
    clear_scale_optimizer_params releases the state when the layer is done.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Indicates whether init_scale_optimizer_params has been called and the layer state is valid.
        self._params_initialized = False

    def init_scale_optimizer_params(
            self,
            layer: nn.Module,
            perm: torch.Tensor,
            H: torch.Tensor,
            G: Optional[torch.Tensor] = None) -> None:
        """Store the state needed to update layer's group scales one at a time.

        Called by the error-correction code before it starts quantizing the layer.
        Records the Hessians, the quantization grid, and perm (the group-aware
        column order the caller is using) so later single_group_update calls can
        map a column index to its group. Must be paired with clear_scale_optimizer_params.

        Expects layer's params to be allocated already: GPTQ and Qronos both call
        allocate_params at the top of single_layer_update.
        """
        if self._params_initialized:
            raise RuntimeError(
                "Scale optimizer params already initialized; call "
                "clear_scale_optimizer_params before initializing a new layer.")
        self._params_initialized = True

        # Extract the quantization configuration from the layer's weight quantizer
        weight_quant = layer.weight_quant

        self.layer = layer
        self.perm = perm
        # Generate unscaled grid in intX or float_eXmY
        self.grid = self.cast_to_solver_dtype_device(
            _build_grid(layer, dtype=H.dtype, device=H.device))
        self.group_size = weight_quant.group_size
        self.allow_negative_s = not weight_quant.restrict_scale_positive and (
            self.grid[-1] != -self.grid[0])
        self.threshold = _grid_threshold(self.grid, weight_quant)

        # Which solver runs depends only on flags fixed at construction, so resolve
        # it once and prepare exactly the matrices it consumes.
        use_diag = self.hessian_mode != 'dense'
        self._group_solver_fn, self._group_strategy = self._solver_family.get_group_solver(
            use_diag, self.per_group_greedy)
        self._prepare_group_matrices(layer, H, G)

    def _prepare_group_matrices(
            self, layer: nn.Module, H: torch.Tensor, G: Optional[torch.Tensor]) -> None:
        """Store the per-group solver's invariant inputs, in its dtype/device.

        Only w_updated changes between groups (error correction rewrites the
        weights as it goes), so everything else is prepared once instead of on
        every single_group_update call:
          - group-diag reads only the Hessian diagonals, so a diagonal/identity H
            is never materialized as a dense matrix;
          - group-greedy reads H/G/W in full, so casting them once avoids
            re-transferring the whole Hessian for every group;
          - group-dense only ever reads its own block, so H/G are left where they
            are and sliced before casting, keeping the per-group transfer small.
        """
        self.H = self.G = self.W = None
        self.H_diag = self.G_diag = None

        if self._group_strategy == 'group-diag':
            if self.hessian_mode == 'identity':
                self.H_diag = torch.ones(
                    H.shape[-1], dtype=self.solver_dtype, device=self.solver_device)
            else:
                self.H_diag = self.cast_to_solver_dtype_device(H.diag())
                self.G_diag = self.cast_to_solver_dtype_device(G.diag()) if G is not None else None
        elif self._group_strategy == 'group-greedy':
            self.H = self.cast_to_solver_dtype_device(H)
            self.G = self.cast_to_solver_dtype_device(G)
            # clone() is required because error correction modifies layer.weight in-place
            W = layer.weight_orig if hasattr(layer,
                                             'weight_orig') else layer.weight.detach().clone()
            self.W = self.cast_to_solver_dtype_device(W)
        else:
            self.H = H
            self.G = G

    def clear_scale_optimizer_params(self) -> None:
        """Release the per-layer state recorded by init_scale_optimizer_params.

        Idempotent: safe to call even if no layer is set, so callers can invoke it
        unconditionally (e.g. on error paths) without tracking state.
        """
        if not self._params_initialized:
            return
        self._params_initialized = False

        del self.layer
        del self.perm
        del self.grid
        del self.group_size
        del self.allow_negative_s
        del self.threshold
        del self._group_solver_fn
        del self._group_strategy
        del self.H
        del self.G
        del self.W
        del self.H_diag
        del self.G_diag

    def should_update_group(self, column_idx: int) -> bool:
        """A group scale update is due only while a layer is being processed and
        column_idx is the first column of a group. self.group_size is only read
        once a layer is set, so the _params_initialized check must short-circuit."""
        return self._params_initialized and column_idx % self.group_size == 0

    def single_group_update(self, incoming_idx: int) -> None:
        """Optimize and write the scale of the group that owns column incoming_idx.

        incoming_idx is an index into the caller's permuted column order; it is
        mapped back through perm to the group it belongs to. Picks the matching
        sweep (diagonal, sequential-greedy, or plain per-group) and writes that
        group's scale into the quantizer.
        """
        if not self._params_initialized:
            raise RuntimeError(
                "Scale optimizer params not initialized; call "
                "init_scale_optimizer_params before single_group_update.")
        incoming_group_idx = self.perm[incoming_idx] // self.group_size
        start = incoming_group_idx * self.group_size
        end = start + self.group_size

        optimal_s = self._bind_group_solver(start, end, incoming_group_idx)()

        # set the optimal s of this group in the QuantLayer
        _set_base_scale(
            self.layer, optimal_s * self.threshold, index=(slice(None), incoming_group_idx))

    def _bind_group_solver(self, start: int, end: int, group_idx: int) -> Callable:
        """Bind this group's arguments into a callable taking no further input.

        The counterpart of _bind_solver_matrices for the group-wise updater:
        _prepare_group_matrices already holds everything invariant, so the only
        tensor read here is the current (error-corrected) weight.
        """
        bound = partial(
            self._group_solver_fn,
            unscaled_grid=self.grid,
            eps=self.tolerance,
            allow_negative_s=self.allow_negative_s)

        if self._group_strategy == 'group-diag':
            # H is diagonal → no cross-group correlations → use fast O(DL) algorithm
            return partial(
                bound,
                w_b=self.cast_to_solver_dtype_device(self.layer.weight[:, start:end]),
                H_diag=self.H_diag[start:end],
                G_diag=None if self.G_diag is None else self.G_diag[start:end])
        if self._group_strategy == 'group-greedy':
            # optimize single group taking into account previous groups
            return partial(
                bound,
                w_orig=self.W,
                w_updated=self.cast_to_solver_dtype_device(self.layer.weight),
                H=self.H,
                group_size=self.group_size,
                group_idx=group_idx,
                G=self.G)
        # optimize just the current group; slicing before the cast keeps the
        # per-group transfer to one block instead of the whole Hessian
        return partial(
            bound,
            w_b=self.cast_to_solver_dtype_device(self.layer.weight[:, start:end]),
            H=self.cast_to_solver_dtype_device(self.H[start:end, start:end]),
            G=None if self.G is None else self.cast_to_solver_dtype_device(
                self.G[start:end, start:end]))


class PiSOLayerHandler():
    """Per-layer forward hook that accumulates the Hessians PiSO needs.

    One is attached to each supported layer by optimize_scale_mode. As calibration
    data flows through, update_batch incrementally builds the Hessians H and G
    from the layer inputs. Once calibration is done, single_layer_update hands
    them to the ScaleOptimizer to compute and write the scales.
    """

    def __init__(
            self,
            layer: nn.Module,
            name: str,
            single_forward: bool,
            matrix_dtype: torch.dtype,
            scale_optimizer: ScaleOptimizer) -> None:

        self.layer = layer
        self.name = name
        self.scale_optimizer = scale_optimizer

        self.single_forward = single_forward
        self.matrix_dtype = matrix_dtype

        dim = self.layer.weight.shape[1]
        self.H = torch.zeros((1, dim, dim),
                             device=self.layer.weight.device,
                             dtype=self.matrix_dtype)
        self.nsamples = 0
        self.nsamples_G = 0
        self.G = torch.zeros_like(self.H) if self.scale_optimizer.cross_act_objective else None
        self.quant_input = None

    def compute_iterative_XtX(self, module, input, current_layer):
        # Accumulate H = X_tilde^T X_tilde as a running mean over batches. For the
        # qronos objective we also stash this batch's input to pair with the float
        # input in compute_iterative_G.
        current_layer.layer_names.add(self.name)

        inp_processed = self.process_input(input)
        if self.scale_optimizer.cross_act_objective:
            if self.quant_input is not None:
                raise RuntimeError(
                    "Expected quant_input to be consumed by compute_iterative_G before "
                    "the next XtX accumulation; got a stale quant_input.")
            self.quant_input = inp_processed.to(dtype=self.H.dtype, copy=True)

        batch_size = inp_processed.shape[-1]

        self.H *= self.nsamples / (self.nsamples + batch_size)

        self.nsamples += batch_size
        inp_processed = math.sqrt(2 / self.nsamples) * inp_processed.to(dtype=self.H.dtype)

        self.H += inp_processed.bmm(inp_processed.transpose(2, 1))

    def compute_iterative_G(self, module, input, current_layer):
        # Accumulate the cross term G = X_tilde^T X (quant input from the previous
        # pass times this pass's float input), for the qronos objective only.
        inp_processed = self.process_input(input)
        batch_size = inp_processed.shape[-1]

        self.G *= self.nsamples_G / (self.nsamples_G + batch_size)

        self.nsamples_G += batch_size
        inp_processed = math.sqrt(2 / self.nsamples) * inp_processed.to(dtype=self.H.dtype)
        self.quant_input = math.sqrt(2 / self.nsamples) * self.quant_input.to(dtype=self.H.dtype)

        self.G += self.quant_input.bmm(inp_processed.transpose(2, 1))

        del self.quant_input
        self.quant_input = None

    def process_input(self, inp):
        # PiSO only supports Linear layers (see ScaleOptimizer.is_layer_valid), so
        # groups is always 1. The returned quant_metadata is unused here because
        # the scale sweep recomputes everything from H/G.
        inp_processed, _ = process_layer_input(self.layer, inp, groups=1)
        return inp_processed

    def update_batch(self, module, input, current_layer):
        is_quant_enabled = module.weight_quant.is_quant_enabled

        if self.scale_optimizer.cross_act_objective and not is_quant_enabled:
            self.compute_iterative_G(module, input, current_layer)
        else:
            self.compute_iterative_XtX(module, input, current_layer)

        if not self.single_forward:
            raise StopFwdException

    def single_layer_update(self):
        # ScaleOptimizer.single_layer_update owns the allocate/offload pair here.
        self.scale_optimizer.single_layer_update(layer=self.layer, H=self.H, G=self.G)
        del self.H
        if self.scale_optimizer.cross_act_objective:
            del self.G


class optimize_scale_mode(layerwise_hook_mode):
    """Context manager that runs standalone (decoupled) PiSO over a whole model.

    On entry it selects the supported layers, attaches a PiSOLayerHandler hook to
    each, and swaps in a forward that catches the early-stop exception. The caller
    feeds calibration data through the model to accumulate Hessians, then calls
    update to optimize every layer's scales. Quantization status is toggled per
    use_quant_activations / quantize_prev so the accumulated activations match
    the chosen reconstruction objective. This is the standalone path; the interleaved
    path instead passes a ScaleOptimizer into GPTQ/Qronos.
    """

    def __init__(
            self,
            model,
            use_quant_activations: bool = True,
            quantize_prev: bool = False,
            single_forward: bool = False,
            solver_dtype: torch.dtype = torch.float32,
            matrix_dtype: torch.dtype = torch.float32,
            tolerance: float = 1e-5,
            solver_batch_size: Optional[int] = None,
            cross_act_objective: bool = False,
            per_group_greedy: bool = False,
            per_group_greedy_reorder: bool = True,
            hessian_mode: str = 'dense',
            solver: str = 'piso') -> None:

        self.scale_optimizer = ScaleOptimizer(
            solver_dtype=solver_dtype,
            solver_device='cuda' if torch.cuda.is_available() else 'cpu',
            tolerance=tolerance,
            solver_batch_size=solver_batch_size,
            cross_act_objective=cross_act_objective,
            per_group_greedy=per_group_greedy,
            per_group_greedy_reorder=per_group_greedy_reorder,
            hessian_mode=hessian_mode,
            solver=solver)

        self.use_quant_activations = use_quant_activations
        self.quantize_prev = quantize_prev
        self.single_forward = single_forward
        self.matrix_dtype = matrix_dtype
        # Weight quant temporarily disabled per layer until its scales are computed;
        # populated on __enter__ and reactivated on __exit__.
        self.modules_to_reactivate_after = dict()

        # Validate attributes received as arguments before touching the model
        self._validate_args()

        super().__init__(
            model=model,
            disable_act_quant=not use_quant_activations,
            disable_bias_quant=not use_quant_activations,
            swap_forward=not single_forward,
        )

    def _is_module_supported(self, module) -> bool:
        return ScaleOptimizer.is_layer_valid(module)

    def _post_enter(self, dict_of_layers) -> None:
        if not (self.quantize_prev or self.single_forward):
            # dict_of_layers maps group name -> [(name, module)]; PiSO uses no
            # parallel groups, so each list holds a single member.
            self.modules_to_reactivate_after = {
                name: members[0][1] for name, members in dict_of_layers.items()}

    def _post_exit(self) -> None:
        # Reactivate weight quantization if it was disabled
        for layer in self.modules_to_reactivate_after.values():
            QuantizationStatusManager.enable_weight_quantization(model=layer, is_training=False)

    def _after_layer_update(self, name: str) -> None:
        if name in self.modules_to_reactivate_after:
            QuantizationStatusManager.disable_weight_quantization(
                model=self.modules_to_reactivate_after[name], is_training=False)

    def catch_stopfwd(self, *args, **kwargs):
        try:
            self.orig_forward(*args, **kwargs)
        except StopFwdException:
            pass
        if self.scale_optimizer.cross_act_objective:
            with quantization_status_manager(
                    self.model,
                    disable_act_quant=True,
                    disable_weight_quant=True,
                    disable_bias_quant=True,
                    is_training=False,
            ):
                try:
                    self.orig_forward(*args, **kwargs)
                except StopFwdException:
                    pass

    @torch.no_grad()
    def __enter__(self):
        return super().__enter__()

    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        # len_parallel_layers / create_weight_orig are part of the shared
        # initialize_module_optimizer contract but unused by PiSO.
        return PiSOLayerHandler(
            layer=layer,
            name=name,
            single_forward=self.single_forward,
            matrix_dtype=self.matrix_dtype,
            scale_optimizer=self.scale_optimizer)

    def _validate_args(self) -> None:
        # TODO (jpga) Move this to LLM args maybe? unsure about it, as the class could be used in other wrokflows
        if self.single_forward and self.quantize_prev:
            raise ValueError(
                'single_forward and quantize_prev are incompatible options. Set at least one of the two to False.'
            )
        if not (self.quantize_prev or self.single_forward):
            warnings.warn(
                'Since quantize_prev=False you may consider using single_forward=True to accelerate the algorithm if memory requiriments allow it.'
            )
        if self.scale_optimizer.hessian_mode == 'diagonal':
            warnings.warn(
                'hessian_mode="diagonal": only the diagonal of H (and G) is used, so off-diagonal '
                'cross-weight interactions are discarded.')
        elif self.scale_optimizer.hessian_mode == 'identity':
            warnings.warn(
                'hessian_mode="identity": scale optimization is data-free (H=I, G=None, pure weight '
                'MSE). The calibration forward passes still run, but the computed H and G are ignored.'
            )
        if self.scale_optimizer.cross_act_objective and not self.quantize_prev:
            raise ValueError('cross_act_objective requires quantize_prev')


# -----------------------------------------------------------------------------
# Integration with error correction (GPTQ / Qronos).
#
# GPTQ and Qronos expose extension hooks on the base GPxQ class (see
# brevitas.graph.gpxq): _act_order_permutation, _resolve_blocksize,
# _on_error_correction_starts, _on_column_reached and _on_layer_finished. The
# two mixins below override those hooks to add, independently:
#   - group-aware column ordering (GroupAwarePermutationMixin), and
#   - PiSO scale optimization interleaved with error correction (PiSOMixin).
# The concrete classes at the bottom compose them with GPTQ / Qronos, mirroring
# the A2GPTQ(_AXE, GPTQ) pattern already used in brevitas for AXE.
# -----------------------------------------------------------------------------


class GroupAwarePermutationMixin:
    """Order columns group-by-group instead of by plain diag(H).

    Overrides only _act_order_permutation (so it applies solely when act_order is
    on): keep each quantization group's weights contiguous and order groups by
    descending aggregated diag(H) (see compute_group_importance_permutation). Usable on its
    own (group-aware act_order without scale optimization) or combined with
    PiSOMixin for the group-interleaved PiSO integration, where processing every
    group contiguously is required.
    """

    def _act_order_permutation(self, hessian_group: torch.Tensor) -> torch.Tensor:
        perm, _ = compute_group_importance_permutation(
            x=hessian_group.diag(), group_size=self.layer.weight_quant.group_size)
        return perm


class PiSOMixin:
    """Interleave PiSO scale optimization with error correction.

    Holds the ScaleOptimizer (always present for a PiSO class) and overrides the
    scale lifecycle hooks so GPTQ / Qronos stay free of scale-specific branching:
      - _resolve_blocksize: align error-correction blocks with quantization groups;
      - _on_error_correction_starts: optimize the layer scale on the fixed grid
        (layer-level updater) or hand the Hessians to the group-wise updater;
      - _on_column_reached: optimize a group's scale at its first column;
      - _on_layer_finished: release the updater's per-layer state.

    Concrete subclasses provide _prepare_input_scale_optimizer to return the (H, G) matrices in
    the layout the updater expects; G is None for the GPTQ objective.
    """

    def __init__(self, *args, scale_optimizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_optimizer = scale_optimizer

    def _resolve_blocksize(self, num_blocks: int) -> int:
        # Process one quantization group per block so group scales can be
        # optimized at block boundaries; fall back to the default otherwise.
        weight_quant = self.layer.weight_quant
        if weight_quant.is_groupwise:
            return weight_quant.group_size
        return super()._resolve_blocksize(num_blocks)

    def _on_error_correction_starts(self, perm: torch.Tensor) -> None:
        H, G = self._prepare_input_scale_optimizer(perm)
        if isinstance(self.scale_optimizer, ScaleOptimizerGroupInterleaved):
            self.scale_optimizer.init_scale_optimizer_params(layer=self.layer, perm=perm, H=H, G=G)
        else:
            G = None if G is None else G.unsqueeze(0)
            self.scale_optimizer.single_layer_update(
                self.layer, H.unsqueeze(0), G, skip_offload=True)

    def _on_column_reached(self, column_idx: int) -> None:
        if self.scale_optimizer.should_update_group(column_idx):
            self.scale_optimizer.single_group_update(column_idx)

    def _on_layer_finished(self) -> None:
        self.scale_optimizer.clear_scale_optimizer_params()

    def _prepare_input_scale_optimizer(
            self, perm: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (H, G) for the updater, inverse-permuted to the natural order.

        G is None for the GPTQ (self-activation) objective.
        """
        raise NotImplementedError


class _PiSOGPTQMixin(PiSOMixin):
    """PiSOMixin specialized for GPTQ: supplies GPTQ's H handoff (no G)."""

    def _prepare_input_scale_optimizer(
            self, perm: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # GPTQ only keeps H; undo the act_order permutation so H is in the layer's
        # natural column order, which is what the ScaleOptimizer expects.
        if self.groups != 1:
            raise ValueError("Scale optimization inside gptq requires groups == 1")
        inv_perm = torch.empty_like(perm, device=self.H.device)
        inv_perm[perm] = torch.arange(len(perm), device=self.H.device)
        H = self.H.squeeze()[:, inv_perm][inv_perm, :]
        return H, None


class _PiSOQronosMixin(PiSOMixin):
    """PiSOMixin specialized for Qronos: supplies both H and G (cross-activation)."""

    def _prepare_input_scale_optimizer(
            self, perm: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Qronos also tracks G; undo the act_order permutation on both. G is
        # transposed to match the ScaleOptimizer's convention (G = X_tilde^T X).
        if self.groups != 1:
            raise ValueError("scale optimization inside qronos requires groups == 1")
        inv_perm = torch.empty_like(perm, device=self.H.device)
        inv_perm[perm] = torch.arange(len(perm), device=self.H.device)
        H = self.H.squeeze()[:, inv_perm][inv_perm, :]
        G = self.G.squeeze()[:, inv_perm][inv_perm, :].T
        return H, G

    def _step2_block_ranges(self) -> List[Tuple[int, int]]:
        """Column blocks for Qronos step-2+, aligned to quantization groups.

        Group-interleaved PiSO optimizes a group's scale via _on_column_reached
        just before Qronos quantizes that group's weights. For that to be correct
        each error-correction block must cover exactly one group, so the block
        boundaries have to fall on group boundaries. blocksize already equals
        group_size here (PiSOMixin._resolve_blocksize), so we iterate groups from
        column 0 but keep the first block starting at column 1 (column 0 is done in
        Qronos step 1). This reproduces the original group-interleaved partition.

        Only the group-wise updater needs this alignment; a layer-level updater
        optimizes all scales once before error correction and imposes no ordering
        constraint, so it defers to the base (reference) Qronos partition.
        """
        if not isinstance(self.scale_optimizer, ScaleOptimizerGroupInterleaved):
            return super()._step2_block_ranges()
        return [(1 if i1 == 0 else i1, min(i1 + self.blocksize, self.columns))
                for i1 in range(0, self.columns, self.blocksize)]


class PiSOLayerInterleavedGPTQ(_PiSOGPTQMixin, GPTQ):
    """GPTQ with layer-interleaved PiSO (paper "Interleaved layer-wise optimization").

    The layer scale is optimized once, right before the layer's error-correction
    step. Works with per-channel or per-group quantization (in the per-group case a
    single pass computes all group scales); it does not interleave scale updates
    with individual groups, so the plain act_order permutation is used.

    Note: because the scale is fixed before error correction, EC is free to order
    columns however it likes, so group-aware ordering is not required here. If a
    group-aware order were ever wanted alongside layer-wise optimization, one would
    add GroupAwarePermutationMixin to the bases; the paper does not use this combo.
    """


class PiSOGroupInterleavedGPTQ(_PiSOGPTQMixin, GroupAwarePermutationMixin, GPTQ):
    """GPTQ with group-interleaved PiSO (paper "Interleaved group-wise optimization").

    Each group's scale is optimized just before error correction quantizes that
    group's weights, so it requires per-group quantization and a group-aware column
    order (groups processed contiguously).
    """


class PiSOLayerInterleavedQronos(_PiSOQronosMixin, Qronos):
    """Qronos with layer-interleaved PiSO (paper "Interleaved layer-wise optimization").

    See PiSOLayerInterleavedGPTQ; the scale is optimized once per layer before
    Qronos' error correction.
    """


class PiSOGroupInterleavedQronos(_PiSOQronosMixin, GroupAwarePermutationMixin, Qronos):
    """Qronos with group-interleaved PiSO (paper "Interleaved group-wise optimization").

    See PiSOGroupInterleavedGPTQ; each group's scale is optimized just before Qronos
    quantizes that group.
    """


class GroupAwarePermGPTQ(GroupAwarePermutationMixin, GPTQ):
    """GPTQ with group-aware act_order but no scale optimization.

    Keeps each quantization group's weights contiguous and orders groups by
    aggregated diag(H); see the group processing order discussion in the paper.
    """


class GroupAwarePermQronos(GroupAwarePermutationMixin, Qronos):
    """Qronos with group-aware act_order but no scale optimization.

    See GroupAwarePermGPTQ.
    """
