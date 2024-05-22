# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from brevitas.quant.scaled_int import Int8AccumulatorAwareWeightQuant
from brevitas.quant.scaled_int import Int8AccumulatorAwareZeroCenterWeightQuant
from brevitas.quant.scaled_int import Int8WeightPerChannelFloat


def test_import_bnn_pynq():
    from brevitas_examples.bnn_pynq import cnv_1w1a
    from brevitas_examples.bnn_pynq import cnv_1w2a
    from brevitas_examples.bnn_pynq import cnv_2w2a
    from brevitas_examples.bnn_pynq import lfc_1w1a
    from brevitas_examples.bnn_pynq import lfc_1w2a
    from brevitas_examples.bnn_pynq import sfc_1w1a
    from brevitas_examples.bnn_pynq import sfc_1w2a
    from brevitas_examples.bnn_pynq import sfc_2w2a
    from brevitas_examples.bnn_pynq import tfc_1w1a
    from brevitas_examples.bnn_pynq import tfc_1w2a
    from brevitas_examples.bnn_pynq import tfc_2w2a


def test_import_image_classification():
    from brevitas_examples.imagenet_classification import quant_mobilenet_v1_4b
    from brevitas_examples.imagenet_classification import quant_proxylessnas_mobile14_4b
    from brevitas_examples.imagenet_classification import quant_proxylessnas_mobile14_4b5b
    from brevitas_examples.imagenet_classification import quant_proxylessnas_mobile14_hadamard_4b


def test_import_tts():
    from brevitas_examples.text_to_speech import quant_melgan_8b


def test_import_stt():
    from brevitas_examples.speech_to_text import quant_quartznet_perchannelscaling_4b
    from brevitas_examples.speech_to_text import quant_quartznet_perchannelscaling_8b
    from brevitas_examples.speech_to_text import quant_quartznet_pertensorscaling_8b


@pytest.mark.parametrize("upscale_factor", [2, 3, 4])
@pytest.mark.parametrize("num_channels", [1, 3])
@pytest.mark.parametrize(
    "weight_quant",
    [
        Int8WeightPerChannelFloat,
        Int8AccumulatorAwareWeightQuant,
        Int8AccumulatorAwareZeroCenterWeightQuant])
def test_super_resolution_float_and_quant_models_match(upscale_factor, num_channels, weight_quant):
    import brevitas.config as config
    from brevitas_examples.super_resolution.models import float_espcn
    from brevitas_examples.super_resolution.models import quant_espcn
    config.IGNORE_MISSING_KEYS = True
    float_model = float_espcn(upscale_factor, num_channels)
    quant_model = quant_espcn(upscale_factor, num_channels, weight_quant=weight_quant)
    quant_model.load_state_dict(float_model.state_dict())


@pytest.mark.parametrize(
    "weight_quant",
    [
        Int8WeightPerChannelFloat,
        Int8AccumulatorAwareWeightQuant,
        Int8AccumulatorAwareZeroCenterWeightQuant])
def test_image_classification_float_and_quant_models_match(weight_quant):
    import brevitas.config as config
    from brevitas_examples.imagenet_classification.a2q.resnet import float_resnet18
    from brevitas_examples.imagenet_classification.a2q.resnet import quant_resnet18
    config.IGNORE_MISSING_KEYS = True
    float_model = float_resnet18(num_classes=10)
    quant_model = quant_resnet18(num_classes=10, weight_quant=weight_quant)
    quant_model.load_state_dict(float_model.state_dict())
