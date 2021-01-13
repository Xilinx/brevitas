import torch

from brevitas.onnx import enable_debug
from brevitas.onnx import export_finn_onnx
from brevitas_examples.bnn_pynq.models import model_with_cfg

REF_MODEL = 'CNV_2W2A'

def test_debug_finn_onnx_export():
    model, cfg = model_with_cfg(REF_MODEL, pretrained=False)
    model.eval()
    debug_hook = enable_debug(model)
    input_tensor = torch.randn(1, 3, 32, 32)
    export_finn_onnx(model, input_shape=input_tensor.shape, export_path='debug.onnx')
    model(input_tensor)
    assert debug_hook.values