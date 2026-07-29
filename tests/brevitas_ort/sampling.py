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
conditions are re-expressed here as validity predicates) and then draws a reproducible
random subset at collection time. The subset is parametrized with plain
``pytest.mark.parametrize`` so that ``pytest -n`` (xdist) still distributes the sampled
nodes across workers.

Two environment variables control sampling:

* ``BREVITAS_ORT_NUM_SAMPLES`` - number of configurations to draw *per test family*
  (capped at the size of the valid space). Defaults to :data:`DEFAULT_NUM_SAMPLES`.
* ``BREVITAS_ORT_SAMPLE_SEED`` - RNG seed, for reproducibility. Defaults to
  :data:`tests.conftest.SEED`.

Determinism note: every xdist worker imports this module and independently reproduces
the *same* sampled set (sampling depends only on the seed and a deterministically
ordered valid list - never on time, PID or process-global RNG state). This is required
or xdist aborts with a "different tests were collected" error.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Optional

from packaging.version import parse

from brevitas import torch_version
from tests.conftest import SEED as DEFAULT_SEED

from .common import BIT_WIDTHS
from .common import Fp8e4m3OCPWeightPerTensorFloat
from .common import LSTM_QUANTIZERS
from .common import QUANT_WBIOL_IMPL
from .common import ShiftedUint8DynamicActPerTensorFloat
from .common import WBIOL_QUANTIZERS

# Configurations drawn *per test family* (wbiol / avgpool / lstm), capped at each
# family's valid-space size. Chosen to target ~5 minutes on a 4-vCPU GitHub-hosted
# runner (ubuntu/windows/macos-latest, matching ``pytest -n logical``). Calibration
# (4 workers, torch 2.9 CPU): the run cost is dominated by a minority of expensive
# configs, giving a marginal cost of ~1s/config; 268 sampled configs ran in ~4 min
# locally, so 100/family (~228 configs, ~3.3 min local) leaves headroom for slower
# CI machines. Override with BREVITAS_ORT_NUM_SAMPLES.
DEFAULT_NUM_SAMPLES = 100

_TORCH_2_1 = parse('2.1')
_TORCH_2_8 = parse('2.8')

WBIOL_EXPORT_TYPES = ['qcdq', 'qcdq_dynamo', 'qonnx', 'qonnx_dynamo']
AVGPOOL_EXPORT_TYPES = ['qcdq', 'qcdq_dynamo']
LSTM_EXPORT_TYPES = ['qcdq', 'qonnx_opset14']


def num_samples() -> int:
    return int(os.environ.get('BREVITAS_ORT_NUM_SAMPLES', DEFAULT_NUM_SAMPLES))


def sample_seed() -> int:
    return int(os.environ.get('BREVITAS_ORT_SAMPLE_SEED', DEFAULT_SEED))


def _sample(configs: list, seed_offset: int) -> list:
    """Deterministically draw ``num_samples()`` configs from ``configs``.

    ``seed_offset`` differentiates the RNG stream per test family so that, e.g., the
    WBIOL and LSTM samples are not correlated. Selection is capped at the valid-space
    size, so a very large BREVITAS_ORT_NUM_SAMPLES naturally yields the full valid set.
    """
    n = min(num_samples(), len(configs))
    rng = random.Random(sample_seed() + seed_offset)
    return rng.sample(configs, n)


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


# -----------------------------------------------------------------------------
# AvgPool
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


# -----------------------------------------------------------------------------
# Public sampled lists (materialized once at import / collection time)
# -----------------------------------------------------------------------------
WBIOL_VALID = _enumerate_wbiol()
AVGPOOL_VALID = _enumerate_avgpool()
LSTM_VALID = _enumerate_lstm()

WBIOL_CONFIGS = _sample(WBIOL_VALID, seed_offset=0)
AVGPOOL_CONFIGS = _sample(AVGPOOL_VALID, seed_offset=1)
LSTM_CONFIGS = _sample(LSTM_VALID, seed_offset=2)


def report_lines() -> List[str]:
    """Human-readable summary for ``pytest_report_header``."""
    return [
        f'brevitas-ort sampling: seed={sample_seed()} num_samples={num_samples()} '
        f'(torch {torch_version})',
        f'  wbiol:   {len(WBIOL_CONFIGS)}/{len(WBIOL_VALID)} valid configs',
        f'  avgpool: {len(AVGPOOL_CONFIGS)}/{len(AVGPOOL_VALID)} valid configs',
        f'  lstm:    {len(LSTM_CONFIGS)}/{len(LSTM_VALID)} valid configs',]
