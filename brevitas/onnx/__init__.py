from .finn.manager import FINNManager
from .pyxir.dpuv1.manager import DPUv1Manager
from .debug import enable_debug

def export_finn_onnx(*args, **kwargs):
    return FINNManager.export_onnx(*args, **kwargs)


def export_dpuv1_onnx(*args, **kwargs):
    return DPUv1Manager.export_onnx(*args, **kwargs)
