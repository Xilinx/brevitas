# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from: https://github.com/Xilinx/finn/blob/master/tests/brevitas/test_brevitas_fc.py

import numpy as np
from packaging import version
import pytest
from qonnx.core.modelwrapper import ModelWrapper
import qonnx.core.onnx_exec as oxe
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.general import RemoveStaticGraphInputs
from qonnx.transformation.infer_shapes import InferShapes
import torch

from brevitas import torch_version
from brevitas.export import export_qonnx
from brevitas.quant_tensor import QuantTensor
from brevitas_examples.bnn_pynq.models import model_with_cfg

FC_INPUT_SIZE = (1, 1, 28, 28)
CNV_INPUT_SIZE = (1, 3, 32, 32)

MAX_WBITS = 2
MAX_ABITS = 2

ATOL = 1e-3


@pytest.mark.parametrize("size", ["TFC", "SFC", "LFC"])
# weight bits
@pytest.mark.parametrize("wbits", [1, MAX_WBITS])
# act bits
@pytest.mark.parametrize("abits", [1, MAX_ABITS])
# Pretrained
@pytest.mark.parametrize("pretrained", [True, False])
def test_brevitas_fc_onnx_export_and_exec(size, wbits, abits, pretrained):
    if size == "LFC" and wbits == 2 and abits == 2:
        pytest.skip(f"No LFC_{MAX_WBITS}W{MAX_ABITS}A present.")
    if wbits > abits:
        pytest.skip("No wbits > abits cases.")
    nname = f"{size}_{wbits}W{abits}A"
    finn_onnx = nname + ".onnx"
    fc, _ = model_with_cfg(nname.lower(), pretrained=pretrained)
    fc.eval()
    # load a random int test vector
    input = torch.randn(FC_INPUT_SIZE)

    export_qonnx(fc, export_path=finn_onnx, input_t=input, input_names=['input'])
    model = ModelWrapper(finn_onnx)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())

    # run using FINN-based execution
    input_dict = {'input': input.detach().numpy()}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    # do forward pass in PyTorch/Brevitas
    expected = fc.forward(input).detach().numpy()
    assert np.isclose(produced, expected, atol=ATOL).all()


# weight bits
@pytest.mark.parametrize("wbits", [1, MAX_WBITS])
# act bits
@pytest.mark.parametrize("abits", [1, MAX_ABITS])
# Pretrained
@pytest.mark.parametrize("pretrained", [True, False])
def test_brevitas_cnv_onnx_export_and_exec(wbits, abits, pretrained):
    if wbits > abits:
        pytest.skip("No wbits > abits cases.")
    nname = f"CNV_{wbits}W{abits}A"
    finn_onnx = nname + ".onnx"
    cnv, _ = model_with_cfg(nname.lower(), pretrained=pretrained)
    cnv.eval()
    # load a random int test vector
    input = torch.randn(CNV_INPUT_SIZE)

    export_qonnx(cnv, export_path=finn_onnx, input_t=input, input_names=['input'])
    model = ModelWrapper(finn_onnx)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())

    # run using FINN-based execution
    input_dict = {"input": input.detach().numpy()}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    # do forward pass in PyTorch/Brevitas
    expected = cnv(input).detach().numpy()
    assert np.isclose(produced, expected, atol=ATOL).all()
