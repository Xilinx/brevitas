# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Adapted from: https://github.com/Xilinx/finn/blob/master/tests/brevitas/test_brevitas_fc.py

import pytest

import numpy as np
import onnx.numpy_helper as nph
import torch
import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import make_build_dir

import brevitas.onnx as bo
from brevitas_examples.bnn_pynq.models import model_with_cfg

INPUT_SIZE = (1, 1, 28, 28)

# activation: None or DataType
@pytest.mark.parametrize("size", ["TFC", "SFC", "LFC"])
# weight bits
@pytest.mark.parametrize("wbits", [1, 2])
# act bits
@pytest.mark.parametrize("abits", [1, 2])
def test_brevitas_fc_onnx_export_and_exec(size, wbits, abits):
    if size == "LFC" and wbits == 2 and abits == 2:
        pytest.skip("No LFC_W2A2 present.")
    if wbits > abits:
        pytest.skip("No wbits > abits cases.")
    nname = "%s_%dW%dA" % (size.lower(), wbits, abits)
    finn_onnx = "%s.onnx" % nname
    fc, _ = model_with_cfg(nname, pretrained=False)
    bo.export_finn_onnx(fc, INPUT_SIZE, finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    # load a random test vector
    input_tensor = np.random.uniform(-1.0, 1.0, size=INPUT_SIZE).astype(np.float32)
    # run using FINN-based execution
    input_dict = {"0": input_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    # run using PyTorch/Brevitas
    input_tensor = torch.from_numpy(input_tensor).float()
    # do forward pass in PyTorch/Brevitas
    expected = fc.forward(input_tensor).detach().numpy()
    assert np.isclose(produced, expected, atol=1e-3).all()