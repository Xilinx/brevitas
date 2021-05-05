from brevitas import config
from .onnx.finn.manager import FINNManager
from .onnx.standard.manager import StdONNXManager
from .onnx.vitis_ai.pyxir.dpuv1.manager import DPUv1Manager
from .onnx.vitis_ai.pyxir.dpuv2.manager import DPUv2Manager
from .onnx.vitis_ai.pyxir.manager import PyXIRManager
from .onnx.vitis_ai.xir.manager import XIRManager
from .onnx.debug import enable_debug
from .pytorch.manager import PytorchQuantManager


def export_finn_onnx(*args, **kwargs):
    return FINNManager.export(*args, **kwargs)


def export_dpuv1_onnx(*args, **kwargs):
    return DPUv1Manager.export(*args, **kwargs)


def export_dpuv2_onnx(*args, **kwargs):
    return DPUv2Manager.export(*args, **kwargs)


def export_standard_onnx(*args, **kwargs):
    return StdONNXManager.export(*args, **kwargs)


def jit_trace_dpuv1(*args, **kwargs):
    return DPUv1Manager.jit_inference_trace(*args, **kwargs)


def is_ongoing_export():
    return config._ONGOING_EXPORT is not None


def is_ongoing_finn_export():
    return config._ONGOING_EXPORT == FINNManager.target_name


def is_ongoing_stdonnx_export():
    return config._ONGOING_EXPORT == StdONNXManager.target_name


def is_ongoing_pyxir_export():
    if config._ONGOING_EXPORT is not None:
        return PyXIRManager.target_name in config._ONGOING_EXPORT
    else:
        return False


def is_ongoing_pytorch_export():
    return config._ONGOING_EXPORT == PytorchQuantManager.target_name