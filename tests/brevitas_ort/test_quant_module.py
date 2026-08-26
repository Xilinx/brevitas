# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import reduce
from operator import mul

from hypothesis import Phase
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
import pytest
import torch

from ..export_fixture import rm_onnx
from .common import *
from .quant_module_cases import build_avgpool_model
from .quant_module_cases import build_lstm_model
from .quant_module_cases import build_wbiol_model
from .sampling import AVGPOOL_CONFIGS
from .sampling import LSTM_MAX_EXAMPLES
from .sampling import LSTM_SHARDS
from .sampling import WBIOL_MAX_EXAMPLES
from .sampling import WBIOL_SHARDS
from .sampling import WBIOL_SHARD_IDS
from .sampling import lstm_config_st
from .sampling import wbiol_config_st

# Skip Hypothesis' shrinking phase: each example is a full ONNX export + ORT inference
# (~seconds), so shrinking a failure could take many minutes. The failing example is still
# reported and reproducible from the global seed (see tests/conftest.py).
_PHASES = (Phase.explicit, Phase.reuse, Phase.generate)


@pytest.mark.parametrize('impl', WBIOL_SHARDS, ids=WBIOL_SHARD_IDS)
@settings(max_examples=WBIOL_MAX_EXAMPLES, phases=_PHASES, deadline=None)
@given(data=st.data())
def test_ort_wbiol(impl, data):
    config = data.draw(wbiol_config_st(impl))
    model = build_wbiol_model(config)

    if impl.__name__ == 'QuantLinear':
        in_size = (1, IN_CH)
    elif impl.__name__ in ('QuantConv1d', 'QuantConvTranspose1d'):
        in_size = (1, IN_CH, FEATURES)
    elif impl.__name__ in ('QuantConv2d', 'QuantConvTranspose2d'):
        in_size = (1, IN_CH, FEATURES, FEATURES)
    else:
        in_size = (1, IN_CH, FEATURES, FEATURES, FEATURES)

    inp = gen_linspaced_data(reduce(mul, in_size), -1, 1).reshape(in_size)
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    export_name = f'qcdq_qop_export_{config.id}.onnx'
    try:
        try:
            close = is_brevitas_ort_close(
                model,
                inp,
                export_name,
                config.export_type,
                tolerance=INT_TOLERANCE,
                first_output_only=True,
                onnx_opset=config.onnx_opset,
                export_q_weight=config.export_q_weight)
        except AllZeroOutput:
            assume(False)  # reject this example and try another
        assert close
    finally:
        rm_onnx(export_name)


@pytest.mark.parametrize('config', AVGPOOL_CONFIGS, ids=[c.id for c in AVGPOOL_CONFIGS])
def test_ort_avgpool(config):
    model = build_avgpool_model(config)
    in_size = (1, IN_CH, FEATURES, FEATURES)
    inp = gen_linspaced_data(reduce(mul, in_size), -1, 1).reshape(in_size)
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    export_name = f'qcdq_quant_avgpool_{config.id}.onnx'
    try:
        try:
            close = is_brevitas_ort_close(
                model,
                inp,
                export_name,
                config.export_type,
                tolerance=INT_TOLERANCE,
                first_output_only=True)
        except AllZeroOutput:
            pytest.skip("Skip testing against all 0s.")
        assert close
    finally:
        rm_onnx(export_name)


@pytest.mark.parametrize('shard', LSTM_SHARDS)
@settings(max_examples=LSTM_MAX_EXAMPLES, phases=_PHASES, deadline=None)
@given(data=st.data())
def test_ort_lstm(shard, data):
    config = data.draw(lstm_config_st(shard))
    model = build_lstm_model(config)
    in_size = (FEATURES, 1, IN_CH)  # seq, batch, in_size
    inp = gen_linspaced_data(reduce(mul, in_size)).reshape(in_size)
    model.eval()
    export_name = f'lstm_export_{config.id}.onnx'
    try:
        assert is_brevitas_ort_close(
            model, inp, export_name, config.export_type, tolerance=FLOAT_TOLERANCE)
    finally:
        rm_onnx(export_name)
