from platform import system

import pytest
from packaging.version import parse
import numpy as np
import torch
import qonnx.core.onnx_exec as oxe
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.general import RemoveStaticGraphInputs
from qonnx.transformation.double_to_single_float import DoubleToSingleFloat

from brevitas import torch_version
from brevitas_examples.imagenet_classification import quant_mobilenet_v1_4b
from brevitas.export import FINNManager

ort_mac_fail = pytest.mark.skipif(
    torch_version >= parse('1.5.0')
    and system() == 'Darwin',
    reason='Issue with ORT and MobileNet export on MacOS on PyTorch >= 1.5.0')


INPUT_SIZE = (1, 3, 224, 224)
ATOL = 1e-3
SEED = 0


@ort_mac_fail
@pytest.mark.parametrize("pretrained", [True])
def test_mobilenet_v1_4b(pretrained):
    finn_onnx = "mobilenet_v1_4b.onnx"
    mobilenet = quant_mobilenet_v1_4b(pretrained)
    mobilenet.eval()
    #load a random test vector
    np.random.seed(SEED)
    numpy_tensor = np.random.random(size=INPUT_SIZE).astype(np.float32)
    # run using PyTorch/Brevitas
    torch_tensor = torch.from_numpy(numpy_tensor).float()
    # do forward pass in PyTorch/Brevitas
    expected = mobilenet(torch_tensor).detach().numpy()
    FINNManager.export_onnx(mobilenet, INPUT_SIZE, finn_onnx)
    model = ModelWrapper(finn_onnx)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    # run using FINN-based execution
    inp_name = model.graph.input[0].name
    input_dict = {inp_name: numpy_tensor}
    output_dict = oxe.execute_onnx(model, input_dict)
    produced = output_dict[list(output_dict.keys())[0]]
    assert np.isclose(produced, expected, atol=ATOL).all()
