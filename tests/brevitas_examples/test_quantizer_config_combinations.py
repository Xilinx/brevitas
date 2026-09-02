# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Exhaustive sweep over the combinations *allowed by* :class:`QuantizerConfig`.

Unlike the reference-parity suites (``test_quantizer_builder.py`` /
``test_input_quantizer_builder.py``), this module does not compare against any
brevitas reference quantizer. It enumerates the full cartesian product of the
:class:`QuantizerConfig` axes, keeps only the combinations that
:meth:`QuantizerConfig.__post_init__` accepts, and then *builds + runs a forward*
on each -- once through :class:`WeightQuantizerBuilder` (hosted in a
``QuantLinear``) and once through :class:`InputQuantizerBuilder` (hosted in a
``QuantIdentity``). The goal is to characterise which config-valid combinations
are actually buildable / runnable end-to-end and which are not.

Combinations that are known to be unsupported (rejected by a builder's
``validate`` or crashing inside brevitas) are marked ``xfail`` from the
documented :func:`_weight_unsupported` / :func:`_input_unsupported` baselines.
The marks are non-strict, so the suite stays green even where the baseline is
imperfect: after a run, inspect ``XFAIL`` (expected-unsupported), ``XPASS``
(now-supported -> tighten the baseline) and any hard ``FAILED`` (a newly
discovered unsupported combo -> add it to the baseline).

Local-loss param methods (MSE / HQO) require ``BREVITAS_JIT=0``; those cases are
skipped when JIT is enabled.
"""

import itertools
from typing import NamedTuple
from typing import Optional

import pytest
import torch

from brevitas import config
from brevitas.inject.enum import QuantType
from brevitas.inject.enum import RestrictValueType
from brevitas.inject.enum import ScalingImplType
from brevitas.inject.enum import ScalingPerOutputType
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas_examples.common.quantizer_builder import build_quantizer
from brevitas_examples.common.quantizer_builder import FloatFormat
from brevitas_examples.common.quantizer_builder import InputQuantizerBuilder
from brevitas_examples.common.quantizer_builder import ParamMethod
from brevitas_examples.common.quantizer_builder import QuantParamType
from brevitas_examples.common.quantizer_builder import WeightQuantizerBuilder
from brevitas_examples.common.quantizer_builder.core import config_from_flat_args

torch.manual_seed(0)

IN_FEATURES = 32
OUT_FEATURES = 16
GROUP_SIZE = 8
BIT_WIDTH = 8

_LOCAL_LOSS_METHODS = (ParamMethod.MSE, ParamMethod.HQO)

# ---------------------------------------------------------------------------
# Axis value lists. ``format`` is a (label, quant_type, float_format,
# float_quant_format) tuple so the int / float discriminator and the float
# sub-format collapse into a single axis.
# ---------------------------------------------------------------------------
_FORMATS = [
    ("int", QuantType.INT, None, None),
    ("float", QuantType.FP, FloatFormat.FLOAT, "e4m3"),
    ("float_ocp", QuantType.FP, FloatFormat.OCP, "e4m3"),
    ("float_fnuz", QuantType.FP, FloatFormat.FNUZ, "e4m3"),]
_PARAM_TYPES = [QuantParamType.SYM, QuantParamType.ASYM]
_GRANULARITIES = [
    ScalingPerOutputType.TENSOR,
    ScalingPerOutputType.CHANNEL,
    ScalingPerOutputType.GROUP,]
_SCALING_IMPL_TYPES = [
    ScalingImplType.STATS,
    ScalingImplType.PARAMETER_FROM_STATS,
    ScalingImplType.DYNAMIC,
    None,]
_RESTRICT_TYPES = [
    RestrictValueType.FP,
    RestrictValueType.POWER_OF_TWO,
    RestrictValueType.SIGNED_FP,]
_SCALING_PARAM_METHODS = [ParamMethod.STATS, ParamMethod.MSE, ParamMethod.HQO]
_ZERO_POINT_PARAM_METHODS = [None, ParamMethod.STATS, ParamMethod.MSE, ParamMethod.HQO]


class Combo(NamedTuple):
    fmt_label: str
    quant_type: QuantType
    float_format: Optional[FloatFormat]
    float_quant_format: Optional[str]
    quant_param_type: QuantParamType
    scaling_per_output_type: ScalingPerOutputType
    scaling_impl_type: Optional[ScalingImplType]
    restrict_scaling_type: RestrictValueType
    scaling_param_method: ParamMethod
    zero_point_param_method: Optional[ParamMethod]

    @property
    def is_group(self) -> bool:
        return self.scaling_per_output_type == ScalingPerOutputType.GROUP

    @property
    def is_float(self) -> bool:
        return self.quant_type == QuantType.FP

    @property
    def is_asym(self) -> bool:
        return self.quant_param_type == QuantParamType.ASYM

    @property
    def uses_local_loss(self) -> bool:
        return (
            self.scaling_param_method in _LOCAL_LOSS_METHODS or
            self.zero_point_param_method in _LOCAL_LOSS_METHODS)


def _iter_all_combos():
    for fmt, param_type, granularity, scaling_impl, restrict, scale_pm, zp_pm in itertools.product(
            _FORMATS,
            _PARAM_TYPES,
            _GRANULARITIES,
            _SCALING_IMPL_TYPES,
            _RESTRICT_TYPES,
            _SCALING_PARAM_METHODS,
            _ZERO_POINT_PARAM_METHODS):
        fmt_label, quant_type, float_format, float_quant_format = fmt
        yield Combo(
            fmt_label=fmt_label,
            quant_type=quant_type,
            float_format=float_format,
            float_quant_format=float_quant_format,
            quant_param_type=param_type,
            scaling_per_output_type=granularity,
            scaling_impl_type=scaling_impl,
            restrict_scaling_type=restrict,
            scaling_param_method=scale_pm,
            zero_point_param_method=zp_pm)


def _config_valid(combo: Combo) -> bool:
    """True iff ``QuantizerConfig.__post_init__`` accepts this combination."""
    try:
        config_from_flat_args(
            combo.quant_type,
            quant_param_type=combo.quant_param_type,
            bit_width=BIT_WIDTH,
            scaling_impl_type=combo.scaling_impl_type,
            scaling_per_output_type=combo.scaling_per_output_type,
            restrict_scaling_type=combo.restrict_scaling_type,
            scaling_param_method=combo.scaling_param_method,
            zero_point_param_method=combo.zero_point_param_method,
            float_format=combo.float_format,
            float_quant_format=combo.float_quant_format)
        return True
    except ValueError:
        return False


# Only the combinations QuantizerConfig itself allows are swept.
VALID_COMBOS = [combo for combo in _iter_all_combos() if _config_valid(combo)]


def _combo_id(combo: Combo) -> str:
    scaling_impl = combo.scaling_impl_type.name.lower() if combo.scaling_impl_type else "no_scale"
    zp_pm = combo.zero_point_param_method.name.lower() if combo.zero_point_param_method else "none"
    return "-".join([
        combo.fmt_label,
        combo.quant_param_type.name.lower(),
        combo.scaling_per_output_type.name.lower(),
        scaling_impl,
        combo.restrict_scaling_type.name.lower(),
        f"spm_{combo.scaling_param_method.name.lower()}",
        f"zppm_{zp_pm}",])


def _weight_unsupported(combo: Combo) -> Optional[str]:
    # Builder fast-fail: WeightSolverComponent.validate rejects activation-only
    # scale modes (DYNAMIC / no_scale) for weight quantizers.
    if combo.scaling_impl_type in (ScalingImplType.DYNAMIC, None):
        return "Weight quantizers require a static scale (not DYNAMIC / no_scale)."
    return None


def _input_unsupported(combo: Combo) -> Optional[str]:
    # Builder fast-fail: InputZeroPointComponent.validate rejects asymmetric
    # activations with a plain STATS scale (a parameter/weight scaling mode; no
    # reference activation quantizer uses it).
    if combo.is_asym and combo.scaling_impl_type == ScalingImplType.STATS:
        return "Asymmetric activations require a static or dynamic scale (not STATS)."
    return None


def _params(unsupported_fn):
    params = []
    for combo in VALID_COMBOS:
        reason = unsupported_fn(combo)
        marks = (pytest.mark.xfail(reason=reason, strict=False),) if reason else ()
        params.append(pytest.param(combo, id=_combo_id(combo), marks=marks))
    return params


def _skip_if_jit_and_local_loss(combo: Combo) -> None:
    if config.JIT_ENABLED and combo.uses_local_loss:
        pytest.skip("Local loss param methods (MSE, HQO) require JIT to be disabled")


def _build_injector(builder_cls, combo: Combo):
    kwargs = {}
    if combo.is_group:
        kwargs["group_size"] = GROUP_SIZE
    return build_quantizer(
        builder_cls,
        combo.quant_type,
        quant_param_type=combo.quant_param_type,
        bit_width=BIT_WIDTH,
        scaling_impl_type=combo.scaling_impl_type,
        scaling_per_output_type=combo.scaling_per_output_type,
        restrict_scaling_type=combo.restrict_scaling_type,
        scaling_param_method=combo.scaling_param_method,
        zero_point_param_method=combo.zero_point_param_method,
        float_format=combo.float_format,
        float_quant_format=combo.float_quant_format,
        kwargs=kwargs).build_quant_injector()


def _apply_input_granularity(injector, combo: Combo):
    if combo.scaling_per_output_type == ScalingPerOutputType.CHANNEL:
        # Per-channel activations carry two different reduction semantics:
        #   * dynamic  -> per-token (per-row): the scale is recomputed per-forward
        #     from the feature dim, reshaped via dynamic_scaling_broadcastable_fn
        #     (the static stats_output_shape is never used).
        #   * static (STATS / PARAMETER_FROM_STATS) -> per-feature: the runtime
        #     stats path reduces over the batch/token dim (stats_reduce_dim=0) and
        #     reshapes to the per-feature scaling shape (1, IN_FEATURES). Using the
        #     per-token reduce dim here collapses the features and makes the
        #     stats.view(stats_output_shape) reshape fail.
        if combo.scaling_impl_type == ScalingImplType.DYNAMIC:
            return injector.let(
                dynamic_scaling_broadcastable_fn=lambda x,
                shape: x.view(*shape[:-1], 1),
                permute_dims=None,
                stats_reduce_dim=1,
                per_channel_broadcastable_shape=(1, IN_FEATURES))
        return injector.let(
            permute_dims=None,
            stats_reduce_dim=0,
            per_channel_broadcastable_shape=(1, IN_FEATURES))
    if combo.scaling_per_output_type == ScalingPerOutputType.GROUP:
        return injector.let(group_dim=-1, group_size=GROUP_SIZE)
    return injector


@pytest.mark.parametrize("combo", _params(_weight_unsupported))
def test_weight_combination_forward(combo: Combo):
    _skip_if_jit_and_local_loss(combo)

    injector = _build_injector(WeightQuantizerBuilder, combo)
    layer_kwargs = {"weight_group_size": GROUP_SIZE} if combo.is_group else {}
    linear = QuantLinear(
        IN_FEATURES,
        OUT_FEATURES,
        bias=False,
        weight_quant=injector,
        return_quant_tensor=False,
        **layer_kwargs)
    linear.eval()
    # Forward triggers lazy scale/zero-point initialization; quant_weight()
    # forces the full quantization path.
    linear(torch.randn(1, IN_FEATURES))
    linear.quant_weight()


@pytest.mark.parametrize("combo", _params(_input_unsupported))
def test_input_combination_forward(combo: Combo):
    _skip_if_jit_and_local_loss(combo)

    injector = _build_injector(InputQuantizerBuilder, combo)
    injector = _apply_input_granularity(injector, combo)
    act = QuantIdentity(act_quant=injector, return_quant_tensor=True)
    x = torch.randn(8, IN_FEATURES)
    # Static scales learn from runtime stats collected in train mode; dynamic /
    # no_scale are unaffected by the extra pass.
    act.train()
    act(x)
    act.eval()
    act(x)
