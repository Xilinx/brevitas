# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import json
import logging
import os

import pytest

from brevitas_examples.bnn_pynq.bnn_pynq_train import launch

# Number of samples to evaluate. With the default batch size of 100 this
# corresponds to 5 batches, i.e. the smallest subset used to generate the
# reference values.
NUM_SAMPLES = 500

# Local file storing the reference Prec@1 values, generated with the highest
# torch version tested in .github/workflows/examples_pytest.yml (2.7.1) and
# checked against all the other versions.
REFERENCE_PATH = os.path.join(os.path.dirname(__file__), 'pretrained_accuracy_reference.json')

# Set BREVITAS_GENERATE_REFERENCE=1 to (re)generate REFERENCE_PATH instead of
# asserting against it.
GENERATE_REFERENCE = bool(int(os.environ.get('BREVITAS_GENERATE_REFERENCE', '0')))


def extract_prec_list(text):
    return [
        l[l.index('Prec@1'):l.index('Prec@5')].rstrip()
        for l in text.splitlines()
        if 'Prec@1' in l and 'Prec@5' in l]


def load_reference():
    if not os.path.exists(REFERENCE_PATH):
        return {}
    with open(REFERENCE_PATH, 'r') as f:
        return json.load(f)


def save_reference_entry(network, log_list):
    # Write per-network files to avoid read-modify-write races under pytest-xdist.
    # These are merged into REFERENCE_PATH after generation.
    path = os.path.join(
        os.path.dirname(REFERENCE_PATH), f'pretrained_accuracy_reference_{network}.json')
    with open(path, 'w') as f:
        json.dump({network: log_list}, f, indent=4, sort_keys=True)
        f.write('\n')


@pytest.mark.parametrize("model", ["TFC", "SFC", "LFC", "CNV"])
@pytest.mark.parametrize("weight_bit_width", [1, 2])
@pytest.mark.parametrize("act_bit_width", [1, 2])
def test_bnn_pynq_pretrained_accuracy(
        bnn_pynq_datasets, caplog, model, weight_bit_width, act_bit_width):
    if model == "LFC" and weight_bit_width == 2 and act_bit_width == 2:
        pytest.skip("No pretrained LFC_W2A2 present.")
    if weight_bit_width > act_bit_width:
        pytest.skip("No weight_bit_width > act_bit_width cases.")

    caplog.set_level(logging.INFO)
    network = f"{model}_{weight_bit_width}W{act_bit_width}A"
    launch([
        '--pretrained',
        '--network',
        network,
        '--evaluate',
        '--gpus',
        'None',
        '--num_samples',
        str(NUM_SAMPLES)])
    log_list = extract_prec_list(caplog.text)

    if GENERATE_REFERENCE:
        save_reference_entry(network, log_list)
    else:
        reference = load_reference()
        assert network in reference, (
            f"No reference value for {network} in {REFERENCE_PATH}. "
            "Generate it with BREVITAS_GENERATE_REFERENCE=1.")
        assert log_list == reference[network]
