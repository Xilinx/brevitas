# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


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
