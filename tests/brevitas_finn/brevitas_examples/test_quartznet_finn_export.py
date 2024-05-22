# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.general import RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
import torch

from brevitas.export import export_qonnx
from brevitas_examples.speech_to_text import quant_quartznet_perchannelscaling_4b

QUARTZNET_POSTPROCESSED_INPUT_SIZE = (1, 64, 256)  # B, features, sequence
MIN_INP_VAL = 0.0
MAX_INP_VAL = 200.0
ATOL = 1e-3


@pytest.mark.parametrize("pretrained", [True, False])
def test_quartznet_asr_4b(pretrained):
    finn_onnx = "quant_quartznet_perchannelscaling_4b.onnx"
    quartznet = quant_quartznet_perchannelscaling_4b(pretrained, export_mode=True)
    quartznet.eval()
    export_qonnx(quartznet, input_shape=QUARTZNET_POSTPROCESSED_INPUT_SIZE, export_path=finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    #load a random test vector
    input_tensor = np.random.uniform(
        MIN_INP_VAL, MAX_INP_VAL, size=QUARTZNET_POSTPROCESSED_INPUT_SIZE).astype(np.float32)
    # run using FINN-based execution
    input_dict = {"0": input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    # run using PyTorch/Brevitas
    input_tensor = torch.from_numpy(input_tensor).float()
    # do forward pass in PyTorch/Brevitas
    expected = quartznet(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=ATOL).all()
