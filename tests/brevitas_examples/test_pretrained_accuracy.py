# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import logging
from urllib import request

import pytest

from brevitas_examples.bnn_pynq.bnn_pynq_train import launch
from brevitas_examples.bnn_pynq.models import get_model_cfg
from tests.marker import requires_pt_lt


@requires_pt_lt('1.6.0')
@pytest.mark.parametrize("model", ["TFC", "SFC", "LFC", "CNV"])
@pytest.mark.parametrize("weight_bit_width", [1, 2])
@pytest.mark.parametrize("act_bit_width", [1, 2])
def test_bnn_pynq_pretrained_accuracy(caplog, model, weight_bit_width, act_bit_width):
    if model == "LFC" and weight_bit_width == 2 and act_bit_width == 2:
        pytest.skip("No pretrained LFC_W2A2 present.")
    if weight_bit_width > act_bit_width:
        pytest.skip("No weight_bit_width > act_bit_width cases.")

    caplog.set_level(logging.INFO)
    network = f"{model}_{weight_bit_width}W{act_bit_width}A"
    cfg = get_model_cfg(network)
    eval_log_url = cfg.get('MODEL', 'EVAL_LOG')
    launch(['--pretrained', '--network', network, '--evaluate', '--gpus', 'None'])
    with request.urlopen(eval_log_url) as r:
        log_list = [
            l[l.index('Prec@1'):l.index('Prec@5')].rstrip() for l in caplog.text.splitlines()]
        reference_prec_list = [
            l[l.index('Prec@1'):l.index('Prec@5')].rstrip()
            for l in r.read().decode('utf-8').splitlines()]
        assert log_list == reference_prec_list
