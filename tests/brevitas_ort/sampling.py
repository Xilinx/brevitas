# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Seeded random sampling of the ORT integration test configuration space.

Historically the ORT integration tests enumerated the full cross-product of every
axis (rounding, layer impl, bit-widths, quantizer, export type, ...) via
``pytest_cases`` and then discarded the vast majority of the ~170k resulting nodes
with ``pytest.skip``. Only a few thousand configurations actually run an ONNX export
plus ORT inference, but generating and collecting all of them is what makes the suite
take over an hour.

This module instead enumerates *only the valid* configuration space (the ``pytest.skip``
conditions are re-expressed here as validity predicates) and draws a reproducible random
subset at collection time. The subset is parametrized with plain
``pytest.mark.parametrize`` so that ``pytest -n`` (xdist) still distributes the sampled
nodes across workers.

Uniform random sampling covers common feature combinations well but is structurally
poor at *rare* buckets: e.g. fp8, dynamic-activation, floor-rounding and the dynamo-QCDQ
export each occupy <0.4% of the WBIOL space, so a single uniform draw of a few hundred
configs will typically contain *zero* of them. To avoid silently skipping whole features
each run, we therefore combine two strategies per test family:

* **Guaranteed rare buckets** - for each rare feature, seeded-sample up to
  :data:`RARE_CAP` configs so the feature is always exercised (~RARE_CAP times).
* **Plain-random common pool** - sample the (large) remaining space uniformly.

The two are unioned/de-duplicated and sorted deterministically.

Environment variables:

* ``BREVITAS_ORT_SAMPLE_SEED`` - RNG seed (default :data:`tests.conftest.SEED`). CI sets
  this per matrix job so different (python, pytorch, platform) jobs test different
  subsets, accumulating combinatorial coverage over time.
* ``BREVITAS_ORT_NUM_SAMPLES`` - optional single override for the *common-pool* count of
  the wbiol and lstm families (rare-bucket guarantees always apply regardless). Unset
  ->  per-family defaults.

Determinism note: every xdist worker imports this module and independently reproduces
the *same* sampled set (sampling depends only on the seed and deterministically ordered
valid lists - never on time, PID or process-global RNG state). This is required or xdist
aborts with a "different tests were collected" error.
"""

import os
import random
from dataclasses import dataclass
from typing import Callable, List, Optional

from packaging.version import parse

from brevitas import torch_version
from tests.conftest import SEED as DEFAULT_SEED

from .common import BIT_WIDTHS
from .common import Fp8e4m3OCPWeightPerTensorFloat
from .common import LSTM_QUANTIZERS
from .common import QUANT_WBIOL_IMPL
from .common import ShiftedUint8DynamicActPerTensorFloat
from .common import WBIOL_QUANTIZERS

# Per-family common-pool sizes. Chosen (together with RARE_CAP) to trade coverage against
# runtime on a 4-vCPU GitHub-hosted runner (matching ``pytest -n logical``). Override the
# common-pool count via BREVITAS_ORT_NUM_SAMPLES.
DEFAULT_WBIOL_SAMPLES = 250
DEFAULT_LSTM_SAMPLES = 100
# Max configs drawn from each rare bucket, guaranteeing every feature is exercised.
RARE_CAP = 15

_TORCH_2_1 = parse('2.1')
_TORCH_2_8 = parse('2.8')

WBIOL_EXPORT_TYPES = ['qcdq', 'qcdq_dynamo', 'qonnx', 'qonnx_dynamo']
AVGPOOL_EXPORT_TYPES = ['qcdq', 'qcdq_dynamo']
LSTM_EXPORT_TYPES = ['qcdq', 'qonnx_opset14']


def sample_seed() -> int:
    return int(os.environ.get('BREVITAS_ORT_SAMPLE_SEED', DEFAULT_SEED))


def _num_samples_override() -> Optional[int]:
    val = os.environ.get('BREVITAS_ORT_NUM_SAMPLES')
    return int(val) if val is not None else None


def wbiol_common_count() -> int:
    override = _num_samples_override()
    return override if override is not None else DEFAULT_WBIOL_SAMPLES


def lstm_common_count() -> int:
    override = _num_samples_override()
    return override if override is not None else DEFAULT_LSTM_SAMPLES


def _sample(configs: list, count: int, seed_offset: int) -> list:
    """Deterministically draw ``min(count, len(configs))`` configs.

    ``seed_offset`` differentiates the RNG stream per family/bucket so their draws are
    uncorrelated. Selection is capped at the pool size, so a very large count naturally
    yields the whole pool.
    """
    n = min(count, len(configs))
    return random.Random(sample_seed() + seed_offset).sample(configs, n)


def _combine(guaranteed: list, common: list, id_fn: Callable) -> list:
    """Union guaranteed + common configs, de-duplicate, and order deterministically."""
    seen = {}
    for cfg in list(guaranteed) + list(common):
        seen[id_fn(cfg)] = cfg
    return [seen[k] for k in sorted(seen)]


# -----------------------------------------------------------------------------
# WBIOL
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class WBIOLConfig:
    quantizer_name: str
    weight_quant: type
    io_quant: type
    output_bit_width: int
    weight_bit_width: int
    input_bit_width: int
    impl: type
    rounding_type: str
    export_type: str

    @property
    def impl_name(self) -> str:
        return self.impl.__name__

    @property
    def is_fp8(self) -> bool:
        return self.weight_quant is Fp8e4m3OCPWeightPerTensorFloat

    @property
    def is_dynamic(self) -> bool:
        return self.io_quant is ShiftedUint8DynamicActPerTensorFloat

    @property
    def onnx_opset(self) -> int:
        return 19 if self.is_fp8 else 18

    @property
    def export_q_weight(self) -> bool:
        # Round weights can be exported as a Q-node (QuantizeLinear); floor weights and
        # A2Q require integer-initializer export instead. FP8 always exports the Q-node.
        if self.is_fp8:
            return True
        return self.rounding_type == 'round' and 'a2q' not in self.quantizer_name

    @property
    def is_rare(self) -> bool:
        return (
            self.is_fp8 or self.is_dynamic or self.rounding_type == 'floor' or
            self.export_type == 'qcdq_dynamo')

    @property
    def id(self) -> str:
        return (
            f'wbiol-{self.quantizer_name}-o{self.output_bit_width}-w{self.weight_bit_width}'
            f'-i{self.input_bit_width}-{self.impl_name}-rtype_{self.rounding_type}'
            f'-{self.export_type}')


def _wbiol_valid(cfg: WBIOLConfig) -> bool:
    quantizer = cfg.quantizer_name
    # FP8 export and FLOOR rounding require all bit-widths equal to 8.
    if cfg.is_fp8 or cfg.rounding_type == 'floor':
        if cfg.weight_bit_width < 8 or cfg.input_bit_width < 8 or cfg.output_bit_width < 8:
            return False
    if cfg.export_type == 'qcdq_dynamo':
        if torch_version < _TORCH_2_8:
            return False
        # Dynamo QCDQ exports weights as a Q-node (round-to-nearest-even only) and does
        # not support quantized-bias / data_ptr export, so it is limited to round +
        # fp8/dynamic quantizers (which don't quantize bias).
        if cfg.rounding_type != 'round':
            return False
        if 'fp8' not in quantizer and 'dynamic' not in quantizer:
            return False
    if cfg.export_type == 'qonnx_dynamo' and torch_version < _TORCH_2_8:
        return False
    if 'per_channel' in quantizer and 'asymmetric' in quantizer:
        return False  # Per-channel zero-point not well supported in ORT.
    if 'QuantLinear' in cfg.impl_name and 'asymmetric' in quantizer:
        return False  # ORT execution unreliable / fails randomly for these.
    if 'dynamic' in quantizer:
        if cfg.output_bit_width != 8 or cfg.input_bit_width != 8:
            return False
        if cfg.export_type not in ('qcdq', 'qcdq_dynamo'):
            return False
    if 'fp8' in quantizer and torch_version < _TORCH_2_1:
        return False
    return True


def _enumerate_wbiol() -> List[WBIOLConfig]:
    configs = []
    for quantizer_name, (weight_quant, io_quant) in WBIOL_QUANTIZERS.items():
        for output_bit_width in BIT_WIDTHS:
            for weight_bit_width in BIT_WIDTHS:
                for input_bit_width in BIT_WIDTHS:
                    for impl in QUANT_WBIOL_IMPL:
                        for rounding_type in ['round', 'floor']:
                            for export_type in WBIOL_EXPORT_TYPES:
                                cfg = WBIOLConfig(
                                    quantizer_name=quantizer_name,
                                    weight_quant=weight_quant,
                                    io_quant=io_quant,
                                    output_bit_width=output_bit_width,
                                    weight_bit_width=weight_bit_width,
                                    input_bit_width=input_bit_width,
                                    impl=impl,
                                    rounding_type=rounding_type,
                                    export_type=export_type)
                                if _wbiol_valid(cfg):
                                    configs.append(cfg)
    return configs


# Rare-feature buckets: (name, predicate, seed offset). A config may belong to several;
# capping each independently guarantees every feature is represented every run.
_WBIOL_RARE_BUCKETS = [
    ('fp8', lambda c: c.is_fp8, 10),
    ('dynamic', lambda c: c.is_dynamic, 11),
    ('floor', lambda c: c.rounding_type == 'floor', 12),
    ('qcdq_dynamo', lambda c: c.export_type == 'qcdq_dynamo', 13),]


def _sample_wbiol(valid: List[WBIOLConfig]) -> List[WBIOLConfig]:
    guaranteed = []
    for _, predicate, offset in _WBIOL_RARE_BUCKETS:
        bucket = [c for c in valid if predicate(c)]
        guaranteed += _sample(bucket, RARE_CAP, offset)
    common = [c for c in valid if not c.is_rare]
    common_sample = _sample(common, wbiol_common_count(), seed_offset=0)
    return _combine(guaranteed, common_sample, id_fn=lambda c: c.id)


# -----------------------------------------------------------------------------
# AvgPool (small enough to always run in full)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AvgPoolConfig:
    input_signed: bool
    output_bit_width: int
    export_type: str

    @property
    def id(self) -> str:
        sign = 'signed' if self.input_signed else 'unsigned'
        return f'avgpool-{sign}-o{self.output_bit_width}-{self.export_type}'


def _avgpool_valid(cfg: AvgPoolConfig) -> bool:
    if cfg.export_type == 'qcdq_dynamo' and torch_version < _TORCH_2_8:
        return False
    return True


def _enumerate_avgpool() -> List[AvgPoolConfig]:
    configs = []
    for input_signed in [True, False]:
        for output_bit_width in BIT_WIDTHS:
            for export_type in AVGPOOL_EXPORT_TYPES:
                cfg = AvgPoolConfig(
                    input_signed=input_signed,
                    output_bit_width=output_bit_width,
                    export_type=export_type)
                if _avgpool_valid(cfg):
                    configs.append(cfg)
    return configs


# -----------------------------------------------------------------------------
# LSTM (float + quant)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LSTMConfig:
    is_quant: bool
    bidirectional: object  # True, False or 'shared_input_hidden'
    cifg: bool
    num_layers: int
    export_type: str
    quantizer_name: Optional[str] = None
    weight_quant: Optional[type] = None
    weight_bit_width: Optional[int] = None

    @property
    def id(self) -> str:
        kind = 'quant_lstm' if self.is_quant else 'float_lstm'
        base = f'{kind}-bidir_{self.bidirectional}-cifg_{self.cifg}-layers_{self.num_layers}'
        if self.is_quant:
            base += f'-{self.quantizer_name}-w{self.weight_bit_width}'
        return f'{base}-{self.export_type}'


def _lstm_valid(cfg: LSTMConfig) -> bool:
    if cfg.is_quant:
        # A2Q doesn't support LSTM export; quantized LSTM can't run on QONNX IR + ORT.
        if cfg.quantizer_name is not None and 'a2q' in cfg.quantizer_name:
            return False
        if cfg.export_type == 'qonnx_opset14':
            return False
    return True


def _enumerate_lstm() -> List[LSTMConfig]:
    configs = []
    bidir_opts = [True, False, 'shared_input_hidden']
    for bidirectional in bidir_opts:
        for cifg in [True, False]:
            for num_layers in [1, 2]:
                for export_type in LSTM_EXPORT_TYPES:
                    cfg = LSTMConfig(
                        is_quant=False,
                        bidirectional=bidirectional,
                        cifg=cifg,
                        num_layers=num_layers,
                        export_type=export_type)
                    if _lstm_valid(cfg):
                        configs.append(cfg)
    for quantizer_name, (weight_quant, _) in LSTM_QUANTIZERS.items():
        for weight_bit_width in BIT_WIDTHS:
            for bidirectional in bidir_opts:
                for cifg in [True, False]:
                    for num_layers in [1, 2]:
                        for export_type in LSTM_EXPORT_TYPES:
                            cfg = LSTMConfig(
                                is_quant=True,
                                bidirectional=bidirectional,
                                cifg=cifg,
                                num_layers=num_layers,
                                export_type=export_type,
                                quantizer_name=quantizer_name,
                                weight_quant=weight_quant,
                                weight_bit_width=weight_bit_width)
                            if _lstm_valid(cfg):
                                configs.append(cfg)
    return configs


def _sample_lstm(valid: List[LSTMConfig]) -> List[LSTMConfig]:
    # Float LSTMs are a small minority (a distinct qonnx_opset14 execution path), so
    # guarantee their representation; sample the large quant pool uniformly.
    float_bucket = [c for c in valid if not c.is_quant]
    quant_pool = [c for c in valid if c.is_quant]
    guaranteed = _sample(float_bucket, RARE_CAP, seed_offset=20)
    common_sample = _sample(quant_pool, lstm_common_count(), seed_offset=2)
    return _combine(guaranteed, common_sample, id_fn=lambda c: c.id)


# -----------------------------------------------------------------------------
# Public sampled lists (materialized once at import / collection time)
# -----------------------------------------------------------------------------
WBIOL_VALID = _enumerate_wbiol()
AVGPOOL_VALID = _enumerate_avgpool()
LSTM_VALID = _enumerate_lstm()

WBIOL_CONFIGS = _sample_wbiol(WBIOL_VALID)
AVGPOOL_CONFIGS = AVGPOOL_VALID  # small enough to run in full
LSTM_CONFIGS = _sample_lstm(LSTM_VALID)


def report_lines() -> List[str]:
    """Human-readable summary for ``pytest_report_header``."""
    n_wbiol_rare = sum(1 for c in WBIOL_CONFIGS if c.is_rare)
    n_lstm_float = sum(1 for c in LSTM_CONFIGS if not c.is_quant)
    override = _num_samples_override()
    knob = f'num_samples={override}' if override is not None else 'num_samples=defaults'
    return [
        f'brevitas-ort sampling: seed={sample_seed()} {knob} rare_cap={RARE_CAP} '
        f'(torch {torch_version})',
        f'  wbiol:   {len(WBIOL_CONFIGS)}/{len(WBIOL_VALID)} valid '
        f'({n_wbiol_rare} guaranteed-rare + {len(WBIOL_CONFIGS) - n_wbiol_rare} common)',
        f'  avgpool: {len(AVGPOOL_CONFIGS)}/{len(AVGPOOL_VALID)} valid (full)',
        f'  lstm:    {len(LSTM_CONFIGS)}/{len(LSTM_VALID)} valid '
        f'({n_lstm_float} guaranteed-float + {len(LSTM_CONFIGS) - n_lstm_float} quant)',]
