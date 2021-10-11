from functools import wraps

from .onnx.finn.manager import FINNManager
from .onnx.generic.manager import BrevitasONNXManager
from .onnx.standard.qoperator.manager import StdQOpONNXManager
from .onnx.vitis_ai.pyxir.manager import PyXIRManager
from .onnx.vitis_ai.xir.manager import XIRManager
from .onnx.debug import enable_debug
from .pytorch.manager import PytorchQuantManager


@wraps(FINNManager.export)
def export_finn_onnx(*args, **kwargs):
    return FINNManager.export(*args, **kwargs)


@wraps(PyXIRManager.export)
def export_pyxir_onnx(*args, **kwargs):
    return PyXIRManager.export(*args, **kwargs)


@wraps(XIRManager.export)
def export_xir(*args, **kwargs):
    return XIRManager.export(*args, **kwargs)


@wraps(BrevitasONNXManager.export)
def export_brevitas_onnx(*args, **kwargs):
    return BrevitasONNXManager.export(*args, **kwargs)


@wraps(StdQOpONNXManager.export)
def export_standard_qop_onnx(*args, **kwargs):
    return StdQOpONNXManager.export(*args, **kwargs)


@wraps(PytorchQuantManager.export)
def export_pytorch_quant(*args, **kwargs):
    return PytorchQuantManager.export(*args, **kwargs)