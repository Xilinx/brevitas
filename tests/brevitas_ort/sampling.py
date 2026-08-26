# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Hypothesis strategies for the ORT integration test configuration space.

Historically the ORT integration tests enumerated the full cross-product of every
axis (rounding, layer impl, bit-widths, quantizer, export type, ...) via
``pytest_cases`` and then discarded the vast majority of the ~170k resulting nodes
with ``pytest.skip``. Only a few thousand configurations actually run an ONNX export
plus ORT inference, but generating and collecting all of them is what makes the suite
take over an hour.

This module instead exposes Hypothesis ``@st.composite`` strategies that draw a *valid*
configuration directly, one axis at a time, so no rejection/``assume`` is needed. Drawing
per-axis (rather than uniformly over a flattened list) means rare features are naturally
well represented - e.g. an fp8 or dynamic-activation quantizer is drawn with probability
~1/n_quantizers rather than <0.4% of the whole space - so no explicit "rare bucket"
handling is required.

The seed governing all generation is configured globally in ``tests/conftest.py``
(:func:`tests.conftest.get_hypothesis_seed`), so these tests need no bespoke seed or
sample-count environment variables. The tests parametrize an outer *shard* axis (e.g. the
layer impl) and draw the remaining axes with Hypothesis; sharding both restores ``pytest -n``
(xdist) parallelism - a single ``@given`` node otherwise runs its examples serially - and
partitions the space into disjoint slices, guaranteeing every shard value is exercised.
"""

from dataclasses import dataclass
from typing import List
from typing import Optional

from hypothesis import strategies as st
from packaging.version import parse

from brevitas import torch_version

from .common import BIT_WIDTHS
from .common import DEFAULT_ONNX_OPSET
from .common import Fp8e4m3OCPWeightPerTensorFloat
from .common import LSTM_QUANTIZERS
from .common import QUANT_WBIOL_IMPL
from .common import QuantLinear
from .common import ShiftedUint8DynamicActPerTensorFloat
from .common import WBIOL_QUANTIZERS

# Hypothesis examples generated *per shard*. Multiplied by the number of shards these give
# the approximate total configurations exercised per test family (see the shard axes below).
WBIOL_MAX_EXAMPLES = 36  # x7 impl shards ~= 250 wbiol configs
LSTM_MAX_EXAMPLES = 15  # x7 (float + 6 quant) shards ~= 105 lstm configs

_TORCH_2_1 = parse('2.1')
_TORCH_2_8 = parse('2.8')
_BIT_WIDTHS = list(BIT_WIDTHS)


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
        return 19 if self.is_fp8 else DEFAULT_ONNX_OPSET

    @property
    def export_q_weight(self) -> bool:
        # Round weights can be exported as a Q-node (QuantizeLinear); floor weights and
        # A2Q require integer-initializer export instead. FP8 always exports the Q-node.
        if self.is_fp8:
            return True
        return self.rounding_type == 'round' and 'a2q' not in self.quantizer_name

    @property
    def id(self) -> str:
        # Includes impl so ONNX export filenames stay unique across shards / xdist workers.
        return (
            f'wbiol-{self.quantizer_name}-o{self.output_bit_width}-w{self.weight_bit_width}'
            f'-i{self.input_bit_width}-{self.impl_name}-rtype_{self.rounding_type}'
            f'-{self.export_type}')


# Impl values are the shard axis for wbiol (one @given node per impl).
WBIOL_SHARDS = list(QUANT_WBIOL_IMPL)
WBIOL_SHARD_IDS = [impl.__name__ for impl in WBIOL_SHARDS]


def _wbiol_quantizers_for(impl) -> List[str]:
    names = list(WBIOL_QUANTIZERS)
    # fp8 requires PyTorch >= 2.1.
    if torch_version < _TORCH_2_1:
        names = [n for n in names if 'fp8' not in n]
    # QuantLinear + asymmetric fails unreliably in ORT execution.
    if impl is QuantLinear:
        names = [n for n in names if 'asymmetric' not in n]
    return names


@st.composite
def wbiol_config_st(draw, impl):
    """Draw a valid WBIOL configuration for a fixed ``impl`` (the shard axis)."""
    quantizer_name = draw(st.sampled_from(_wbiol_quantizers_for(impl)))
    weight_quant, io_quant = WBIOL_QUANTIZERS[quantizer_name]
    is_fp8 = weight_quant is Fp8e4m3OCPWeightPerTensorFloat
    is_dynamic = io_quant is ShiftedUint8DynamicActPerTensorFloat

    rounding_type = draw(st.sampled_from(['round', 'floor']))

    # Bit-widths: fp8 export and floor rounding require all-8; dynamic act quant requires
    # 8-bit input/output. Otherwise draw each independently.
    if is_fp8 or rounding_type == 'floor':
        o = w = i = 8
    elif is_dynamic:
        o, i = 8, 8
        w = draw(st.sampled_from(_BIT_WIDTHS))
    else:
        o = draw(st.sampled_from(_BIT_WIDTHS))
        w = draw(st.sampled_from(_BIT_WIDTHS))
        i = draw(st.sampled_from(_BIT_WIDTHS))

    # Compatible export types.
    exports = ['qcdq', 'qonnx']
    if torch_version >= _TORCH_2_8:
        exports.append('qonnx_dynamo')
        # Dynamo QCDQ exports weights as a Q-node (round only) and cannot export quantized
        # bias, so it is limited to round + fp8/dynamic quantizers (which don't quantize bias).
        if rounding_type == 'round' and (is_fp8 or is_dynamic):
            exports.append('qcdq_dynamo')
    # Dynamic act quant is only supported for the QCDQ export paths.
    if is_dynamic:
        exports = [e for e in exports if e in ('qcdq', 'qcdq_dynamo')]
    export_type = draw(st.sampled_from(exports))

    return WBIOLConfig(
        quantizer_name=quantizer_name,
        weight_quant=weight_quant,
        io_quant=io_quant,
        output_bit_width=o,
        weight_bit_width=w,
        input_bit_width=i,
        impl=impl,
        rounding_type=rounding_type,
        export_type=export_type)


# -----------------------------------------------------------------------------
# AvgPool (small enough to enumerate exhaustively)
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


def _avgpool_export_types() -> List[str]:
    exports = ['qcdq']
    if torch_version >= _TORCH_2_8:
        exports.append('qcdq_dynamo')
    return exports


def enumerate_avgpool() -> List[AvgPoolConfig]:
    # Only 28 valid configs; enumerated exhaustively rather than sampled.
    return [
        AvgPoolConfig(input_signed=s, output_bit_width=b, export_type=e) for s in (True, False)
        for b in BIT_WIDTHS for e in _avgpool_export_types()]


AVGPOOL_CONFIGS = enumerate_avgpool()


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


# Shard axis for lstm: the float model, plus one shard per quant weight quantizer.
LSTM_FLOAT_SHARD = 'float'
LSTM_SHARDS = [LSTM_FLOAT_SHARD] + list(LSTM_QUANTIZERS)
_BIDIR_OPTS = [True, False, 'shared_input_hidden']


@st.composite
def lstm_config_st(draw, shard):
    """Draw a valid LSTM configuration for a fixed shard ('float' or a quantizer name)."""
    bidirectional = draw(st.sampled_from(_BIDIR_OPTS))
    cifg = draw(st.booleans())
    num_layers = draw(st.sampled_from([1, 2]))

    if shard == LSTM_FLOAT_SHARD:
        # Float LSTM can run on both the qonnx (opset14) and qcdq paths.
        export_type = draw(st.sampled_from(['qcdq', 'qonnx_opset14']))
        return LSTMConfig(
            is_quant=False,
            bidirectional=bidirectional,
            cifg=cifg,
            num_layers=num_layers,
            export_type=export_type)

    # Quantized LSTM: only the qcdq path is supported (qonnx opset14 requires the qonnx lib).
    weight_quant, _ = LSTM_QUANTIZERS[shard]
    weight_bit_width = draw(st.sampled_from(_BIT_WIDTHS))
    return LSTMConfig(
        is_quant=True,
        bidirectional=bidirectional,
        cifg=cifg,
        num_layers=num_layers,
        export_type='qcdq',
        quantizer_name=shard,
        weight_quant=weight_quant,
        weight_bit_width=weight_bit_width)
