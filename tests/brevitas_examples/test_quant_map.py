# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests: the static quant-map lookups vs the quantizer-builder equivalents.

``generate_quantizers`` selects its quantizers by indexing the static
``WEIGHT_QUANT_MAP`` / ``INPUT_QUANT_MAP`` with string keys;
:func:`~brevitas_examples.common.quantizer_builder.create_weight_quantizer` and
:func:`~brevitas_examples.common.quantizer_builder.create_input_quantizer`
assemble the equivalent injectors from those same keys through the builder.

Both modules walk every leaf of their map -- so new leaves are picked up
automatically -- build the quantizer both ways and assert the two are equivalent
once hosted by a layer:

  1. identical module hierarchy; and
  2. identical quantized tensors and layer outputs (``torch.equal``).

The axes that are not part of a leaf's identity are supplied identically to both
sides, mirroring the ``.let(...)`` finalization ``generate_quantizers`` performs
on the looked-up leaf: the bit width is left at its default (every leaf is
8-bit), weight group size arrives as the ``weight_group_size`` layer argument,
and the per-row / per-group activation attributes are injected explicitly.

The comparison mirrors ``test_quantizer_builder.py`` and
``test_input_quantizer_builder.py`` because the quantizer injectors themselves
define no ``__eq__``.
"""

import pytest
import torch

from brevitas import config
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas_examples.common.quantizer_builder import create_input_quantizer
from brevitas_examples.common.quantizer_builder import create_weight_quantizer
from tests.brevitas_examples.common import assert_state_dict_parity
from tests.brevitas_examples.common import compare_quant_tensors
from tests.brevitas_examples.common import module_fingerprint
from tests.brevitas_examples.quant_map_reference import INPUT_QUANT_MAP
from tests.brevitas_examples.quant_map_reference import WEIGHT_QUANT_MAP

torch.manual_seed(0)

IN_FEATURES = 32
OUT_FEATURES = 16
GROUP_SIZE = 8


def _iter_leaves(node, path=()):
    """Yield ``(keys, quantizer_class)`` for every leaf of a quant map."""
    if isinstance(node, dict):
        for key, child in node.items():
            yield from _iter_leaves(child, path + (key,))
    else:
        yield path, node


# ---------------------------------------------------------------------------
# WEIGHT_QUANT_MAP
# ---------------------------------------------------------------------------
WEIGHT_LEAVES = list(_iter_leaves(WEIGHT_QUANT_MAP))
WEIGHT_LEAF_IDS = ["-".join(keys) for keys, _ in WEIGHT_LEAVES]

_WEIGHT_XFAIL = {}


def _make_quant_linear(weight_quant, granularity):
    # NOTE: return_quant_tensor must be False here. With weight-only
    # quantization (no input_quant), the layer input stays a plain Tensor, so
    # the layer cannot emit a QuantTensor output and would otherwise raise
    # "QuantLayer is not correctly configured". The quantized weight is still
    # compared directly via ``quant_weight()``.
    # ``weight_group_size`` is forwarded to the injector as ``group_size``.
    layer_kwargs = {'weight_group_size': GROUP_SIZE} if granularity == 'per_group' else {}
    return QuantLinear(
        IN_FEATURES,
        OUT_FEATURES,
        bias=False,
        weight_quant=weight_quant,
        return_quant_tensor=False,
        **layer_kwargs)


def test_weight_map_leaves_collected():
    """Guard against the leaf walk silently emptying the parametrization."""
    assert WEIGHT_LEAVES, "No WEIGHT_QUANT_MAP leaves were collected."
    assert len(set(WEIGHT_LEAF_IDS)) == len(WEIGHT_LEAF_IDS), "Duplicate leaf ids."
    assert all(len(keys) == 5 for keys, _ in WEIGHT_LEAVES), "Expected 5-level map paths."


@pytest.mark.parametrize("keys, ref_quant", WEIGHT_LEAVES, ids=WEIGHT_LEAF_IDS)
def test_create_weight_quantizer_matches_map_leaf(keys, ref_quant):
    _, _, param_method, granularity, _ = keys

    # Local-loss param methods (MSE, HQO) rely on Python control flow during the
    # optimization and require JIT to be disabled.
    if config.JIT_ENABLED and param_method in ('mse', 'hqo'):
        pytest.skip("Local loss param methods (MSE, HQO) require JIT to be disabled")

    if keys in _WEIGHT_XFAIL:
        pytest.xfail(_WEIGHT_XFAIL[keys])

    builder_quant = create_weight_quantizer(*keys)

    ref_linear = _make_quant_linear(ref_quant, granularity)
    builder_linear = _make_quant_linear(builder_quant, granularity)

    # 1) Module hierarchy + scalar attributes must match 1-to-1. Checked before
    # syncing weights so a structural mismatch is reported as a clear diff rather
    # than an opaque "Missing key(s) in state_dict" error.
    assert module_fingerprint(ref_linear) == module_fingerprint(builder_linear)

    # Make both layers carry identical float weights so the only difference that
    # could appear is in the quantization path itself. Only the float weight is
    # copied (not the full state_dict): for MSE / PARAMETER_FROM_STATS scaling
    # the learned scale parameter is excluded from state_dict() until it has
    # been initialized, so a strict load_state_dict would spuriously fail.
    builder_linear.weight.data.copy_(ref_linear.weight.data)

    ref_linear.eval()
    builder_linear.eval()

    # Mock forward pass to trigger lazy initialization of any parameter-based
    # scaling (e.g. MSE / PARAMETER_FROM_STATS), so the learned scales are
    # initialized from the (now identical) weights and are directly comparable.
    mock_input = torch.randn(1, IN_FEATURES)
    ref_linear(mock_input)
    builder_linear(mock_input)

    # 2) The quantized weight tensors themselves must match exactly.
    compare_quant_tensors(ref_linear.quant_weight(), builder_linear.quant_weight())

    # 3) Persistent state (learned scales / zero points / buffers) must match.
    assert_state_dict_parity(ref_linear, builder_linear)

    # 4) Quantized layer output tensors must match exactly. With
    # return_quant_tensor=False the layers return plain Tensors.
    x = torch.randn(1, IN_FEATURES)
    assert torch.equal(ref_linear(x), builder_linear(x))


# ---------------------------------------------------------------------------
# INPUT_QUANT_MAP
# ---------------------------------------------------------------------------
# INPUT_QUANT_MAP is ragged: 'no_scale' is indexed on (format, scale_type,
# quant_type) only, every other scale type on the full six axes.
INPUT_LEAVES = list(_iter_leaves(INPUT_QUANT_MAP))
INPUT_LEAF_IDS = ["-".join(keys) for keys, _ in INPUT_LEAVES]

_INPUT_XFAIL = {}

# The asym MSE init ops (AbsMinMax / NegativeMinOrZero) take dtype/device
# constructor args to build their `zero` buffer. ActQuantSolver provides neither
# (only WeightQuantSolver does, via tracked_parameter_list), so the MSE
# sub-injectors' `(this << 1).dtype/.device` cannot resolve. Supplying
# dtype=None/device=None at the top level satisfies them (the init ops default to
# None anyway). Both the reference and the builder need it, so it is applied to
# both rather than baked into create_input_quantizer.
_INPUT_DTYPE_DEVICE = {
    ('int', 'static', 'float_scale', 'mse', 'per_tensor', 'asym'),}


def _build_input_quant(keys):
    """Call create_input_quantizer with the (ragged) map path."""
    if len(keys) == 3:
        return create_input_quantizer(*keys)
    fmt, scale_type, precision, param_method, granularity, quant_type = keys
    return create_input_quantizer(
        fmt,
        scale_type,
        quant_type,
        input_scale_precision=precision,
        input_param_method=param_method,
        input_quant_granularity=granularity)


# generate_quantizers applies these runtime .let() overrides to the per_row /
# per_group dynamic activation quantizers (they are not baked into the reference
# classes). A bare QuantIdentity cannot auto-resolve the per-channel
# broadcastable shape (per_row) or group_dim (per_group), so those are injected
# manually too. Applied to both sides so the comparison stays fair.
def _apply_granularity_overrides(act_quant, granularity):
    if granularity == "per_row":
        # per_row scale is per output feature of a (N, IN_FEATURES) input.
        return act_quant.let(
            dynamic_scaling_broadcastable_fn=lambda x,
            shape: x.view(*shape[:-1], 1),
            permute_dims=None,
            stats_reduce_dim=1,
            per_channel_broadcastable_shape=(1, IN_FEATURES))
    if granularity == "per_group":
        return act_quant.let(group_dim=-1, group_size=GROUP_SIZE)
    return act_quant


def test_input_map_leaves_collected():
    """Guard against the leaf walk silently emptying the parametrization."""
    assert INPUT_LEAVES, "No INPUT_QUANT_MAP leaves were collected."
    assert len(set(INPUT_LEAF_IDS)) == len(INPUT_LEAF_IDS), "Duplicate leaf ids."
    # The map is ragged: no_scale paths are 3 keys deep, everything else 6.
    assert {len(keys) for keys, _ in INPUT_LEAVES} == {3, 6}
    assert all(keys[1] == 'no_scale' for keys, _ in INPUT_LEAVES if len(keys) == 3)


@pytest.mark.parametrize("keys, ref_quant", INPUT_LEAVES, ids=INPUT_LEAF_IDS)
def test_create_input_quantizer_matches_map_leaf(keys, ref_quant):
    scale_type = keys[1]
    param_method = keys[3] if len(keys) == 6 else 'stats'
    granularity = keys[4] if len(keys) == 6 else 'per_tensor'

    # Local-loss param methods (MSE) require JIT to be disabled.
    if config.JIT_ENABLED and param_method == 'mse':
        pytest.skip("Local loss param methods (MSE) require JIT to be disabled")

    # Per-tensor / per-row dynamic scaling (RuntimeDynamicStatsScaling) takes the
    # broadcastable reshape as a plain Callable, which TorchScript cannot compile.
    # Per-group (RuntimeDynamicGroupStatsScaling) is a ScriptModule and is fine.
    if config.JIT_ENABLED and scale_type == 'dynamic' and granularity != 'per_group':
        pytest.skip("Per-tensor/per-row dynamic act scaling requires JIT to be disabled")

    if keys in _INPUT_XFAIL:
        pytest.xfail(_INPUT_XFAIL[keys])

    builder_quant = _build_input_quant(keys)

    ref_quant = _apply_granularity_overrides(ref_quant, granularity)
    builder_quant = _apply_granularity_overrides(builder_quant, granularity)
    if keys in _INPUT_DTYPE_DEVICE:
        ref_quant = ref_quant.let(dtype=None, device=None)
        builder_quant = builder_quant.let(dtype=None, device=None)

    ref_act = QuantIdentity(act_quant=ref_quant, return_quant_tensor=True)
    builder_act = QuantIdentity(act_quant=builder_quant, return_quant_tensor=True)

    # 1) Module hierarchy + scalar attributes must match 1-to-1.
    assert module_fingerprint(ref_act) == module_fingerprint(builder_act)

    # Collect identical runtime statistics on both, then compare the quantized
    # activations. Static scaling learns its scale from runtime stats, so the
    # same input is run through both in train mode before eval; dynamic scaling
    # is recomputed per-forward and is unaffected by the extra pass.
    x = torch.randn(8, IN_FEATURES)
    for act in (ref_act, builder_act):
        act.train()
        act(x)
        act.eval()

    # 2) The quantized activation tensors must match exactly.
    compare_quant_tensors(ref_act(x), builder_act(x))

    # 3) Persistent state must match.
    assert_state_dict_parity(ref_act, builder_act)
