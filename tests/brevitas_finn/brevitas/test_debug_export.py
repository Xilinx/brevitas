import torch

from brevitas.export import export_finn_onnx, enable_debug
from brevitas_examples import bnn_pynq

REF_MODEL = 'CNV_2W2A'


def test_debug_finn_onnx_export():
    model, cfg = bnn_pynq.model_with_cfg(REF_MODEL, pretrained=False)
    model.eval()
    debug_hook = enable_debug(model)
    input_tensor = torch.randn(1, 3, 32, 32)
    export_finn_onnx(model, input_t=input_tensor, export_path='finn_debug.onnx')
    model(input_tensor)
    assert debug_hook.values