from .onnx.finn.manager import FINNManager
from .onnx.standard.manager import StdONNXManager
from .onnx.pyxir.dpuv1.manager import DPUv1Manager
from .onnx.pyxir.dpuv2.manager import DPUv2Manager
from .onnx.pyxir.manager import PyXIRManager
from .pytorch.manager import PytorchQuantManager


_ONGOING_EXPORT = None


class ExportContext:

    def __init__(self, target):
        self.target = target

    def __enter__(self):
        global _ONGOING_EXPORT
        assert _ONGOING_EXPORT is None
        _ONGOING_EXPORT = self.target

    def __exit__(self, exc_type, exc_value, exc_traceback):
        global _ONGOING_EXPORT
        assert _ONGOING_EXPORT is not None
        _ONGOING_EXPORT = None


def is_ongoing_finn_export():
    return ongoing_export == FINNManager.target_name


def is_ongoing_stdonnx_export():
    return ongoing_export == StdONNXManager.target_name


def is_ongoing_pyxir_export():
    if ongoing_export is not None:
        return PyXIRManager.target_name in ongoing_export
    else:
        return False


def is_ongoing_pytorch_export():
    return ongoing_export == PytorchQuantManager.target_name