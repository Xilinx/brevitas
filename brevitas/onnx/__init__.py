from brevitas.export.onnx.finn.manager import FINNManager
from brevitas.export.onnx.pyxir.dpuv1.manager import DPUv1Manager
from brevitas.export.onnx.pyxir.dpuv2.manager import DPUv2Manager
from brevitas.export.onnx.standard.manager import StdONNXManager
from brevitas.export.onnx.debug import enable_debug


def export_finn_onnx(*args, **kwargs):
    return FINNManager.export_onnx(*args, **kwargs)


def export_dpuv1_onnx(*args, **kwargs):
    return DPUv1Manager.export_onnx(*args, **kwargs)


def export_dpuv2_onnx(*args, **kwargs):
    return DPUv2Manager.export_onnx(*args, **kwargs)


def export_standard_onnx(*args, **kwargs):
    return StdONNXManager.export_onnx(*args, **kwargs)