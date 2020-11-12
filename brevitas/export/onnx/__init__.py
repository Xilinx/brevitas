from .finn.manager import FINNManager
from .pyxir.dpuv1.manager import DPUv1Manager
from .pyxir.dpuv2.manager import DPUv2Manager
from .standard.manager import StdONNXManager
from .debug import enable_debug


def export_finn_onnx(*args, **kwargs):
    return FINNManager.export_onnx(*args, **kwargs)


def export_dpuv1_onnx(*args, **kwargs):
    return DPUv1Manager.export_onnx(*args, **kwargs)


def export_dpuv2_onnx(*args, **kwargs):
    return DPUv2Manager.export_onnx(*args, **kwargs)


def export_standard_onnx(*args, **kwargs):
    return StdONNXManager.export_onnx(*args, **kwargs)


def jit_trace_dpuv1(*args, **kwargs):
    return DPUv1Manager.jit_trace(*args, **kwargs)