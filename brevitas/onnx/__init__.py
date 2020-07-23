from .finn.manager import FINNManager


def export_finn_onnx(*args, **kwargs):
    return FINNManager.export_onnx(*args, **kwargs)

