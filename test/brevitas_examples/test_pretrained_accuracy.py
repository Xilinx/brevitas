import logging
from urllib import request
from common import mnist_datapath_fixture
import pytest
from brevitas_examples.bnn_pynq.bnn_pynq_train import main
from brevitas_examples.bnn_pynq.models import get_model_cfg


@pytest.mark.parametrize("model", ["TFC", "SFC", "LFC"])
@pytest.mark.parametrize("weight_bit_width", [1, 2])
@pytest.mark.parametrize("act_bit_width", [1, 2])
def test_bnn_pynq_fc_pretrained_accuracy(caplog, model, weight_bit_width, act_bit_width, mnist_datapath):
    if model == "LFC" and weight_bit_width == 2 and act_bit_width == 2:
        pytest.skip("No pretrained LFC_W2A2 present.")
    if weight_bit_width > act_bit_width:
        pytest.skip("No weight_bit_width > act_bit_width cases.")

    caplog.set_level(logging.INFO)
    network = f"{model}_{weight_bit_width}W{act_bit_width}A"
    cfg = get_model_cfg(network)
    eval_log_url = cfg.get('MODEL', 'EVAL_LOG')
    main(['--pretrained', '--network', network, '--evaluate', '--gpus', 'None', '--datadir', mnist_datapath])
    with request.urlopen(eval_log_url) as r:
        log_list = [l[l.index('Prec@1'):] for l in caplog.text.splitlines()]
        reference_prec_list = [l[l.index('Prec@1'):] for l in r.read().decode('utf-8').splitlines()]
        assert log_list == reference_prec_list