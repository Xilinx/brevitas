import pytest

import numpy as np
import torch
import finn.core.onnx_exec as oxe
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.general import GiveUniqueNodeNames
from finn.transformation.general import RemoveStaticGraphInputs
from finn.transformation.double_to_single_float import DoubleToSingleFloat

from brevitas.onnx import FINNManager

from tests.common_xfail import check_expected_win_nox_fail


QUARTZNET_POSTPROCESSED_INPUT_SIZE = (1, 64, 256)  # B, features, sequence
MIN_INP_VAL = 0.0
MAX_INP_VAL = 200.0
ATOL = 1e-3


@pytest.mark.parametrize("pretrained", [True, False])
@check_expected_win_nox_fail
def test_quartznet_asr_4b(pretrained):
    # inline import to make xfail work on the import error
    from brevitas_examples.speech_to_text import quant_quartznet_perchannelscaling_4b

    finn_onnx = "quant_quartznet_perchannelscaling_4b.onnx"
    quartznet = quant_quartznet_perchannelscaling_4b(pretrained, export_mode=True)
    quartznet.eval()
    FINNManager.export_onnx(quartznet, QUARTZNET_POSTPROCESSED_INPUT_SIZE, finn_onnx)
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
