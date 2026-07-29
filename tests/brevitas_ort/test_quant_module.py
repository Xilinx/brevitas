# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import reduce
from operator import mul

import pytest
import torch

from ..export_fixture import rm_onnx
from .common import *
from .quant_module_cases import build_avgpool_model
from .quant_module_cases import build_lstm_model
from .quant_module_cases import build_wbiol_model
from .sampling import AVGPOOL_CONFIGS
from .sampling import LSTM_CONFIGS
from .sampling import WBIOL_CONFIGS


@pytest.mark.parametrize('config', WBIOL_CONFIGS, ids=[c.id for c in WBIOL_CONFIGS])
def test_ort_wbiol(config):
    model = build_wbiol_model(config)
    impl = config.impl_name

    if impl in ('QuantLinear'):
        in_size = (1, IN_CH)
    elif impl in ('QuantConv1d', 'QuantConvTranspose1d'):
        in_size = (1, IN_CH, FEATURES)
    elif impl in ('QuantConv2d', 'QuantConvTranspose2d'):
        in_size = (1, IN_CH, FEATURES, FEATURES)
    elif impl in ('QuantConv3d', 'QuantConvTranspose3d'):
        in_size = (1, IN_CH, FEATURES, FEATURES, FEATURES)
    else:
        raise RuntimeError(f"Unsupported operation {impl}")

    inp = gen_linspaced_data(reduce(mul, in_size), -1, 1).reshape(in_size)

    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    export_name = f'qcdq_qop_export_{config.id}.onnx'
    assert is_brevitas_ort_close(
        model,
        inp,
        export_name,
        config.export_type,
        tolerance=INT_TOLERANCE,
        first_output_only=True,
        onnx_opset=config.onnx_opset,
        export_q_weight=config.export_q_weight)

    rm_onnx(export_name)


@pytest.mark.parametrize('config', AVGPOOL_CONFIGS, ids=[c.id for c in AVGPOOL_CONFIGS])
def test_ort_avgpool(config):
    model = build_avgpool_model(config)
    in_size = (1, IN_CH, FEATURES, FEATURES)
    inp = gen_linspaced_data(reduce(mul, in_size), -1, 1).reshape(in_size)
    model(torch.from_numpy(inp))  # accumulate scale factors
    model.eval()
    export_name = f'qcdq_quant_avgpool_{config.id}.onnx'
    assert is_brevitas_ort_close(
        model, inp, export_name, config.export_type, tolerance=INT_TOLERANCE, first_output_only=True)
    rm_onnx(export_name)


@pytest.mark.parametrize('config', LSTM_CONFIGS, ids=[c.id for c in LSTM_CONFIGS])
def test_ort_lstm(config):
    model = build_lstm_model(config)
    in_size = (FEATURES, 1, IN_CH)  # seq, batch, in_size
    inp = gen_linspaced_data(reduce(mul, in_size)).reshape(in_size)
    model.eval()
    export_name = f'lstm_export_{config.id}.onnx'
    assert is_brevitas_ort_close(
        model, inp, export_name, config.export_type, tolerance=FLOAT_TOLERANCE)
    rm_onnx(export_name)
