from .finn.manager import FINNManager
from .debug import enable_debug

def export_finn_onnx(*args, **kwargs):
    return FINNManager.export_onnx(*args, **kwargs)

