from functools import wraps

from .onnx.finn.manager import FINNManager
from .onnx.qonnx.manager import QONNXManager
from .onnx.standard.qoperator.manager import StdQOpONNXManager
from .onnx.standard.qcdq.manager import StdQCDQONNXManager
from .pytorch.manager import PytorchQuantManager
from .onnx.debug import enable_debug


@wraps(FINNManager.export)
def export_finn_onnx(*args, **kwargs):
    return FINNManager.export(*args, **kwargs)


@wraps(QONNXManager.export)
def export_brevitas_onnx(*args, **kwargs):  # alias for qonnx
    return QONNXManager.export(*args, **kwargs)


@wraps(QONNXManager.export)
def export_qonnx(*args, **kwargs):  
    return QONNXManager.export(*args, **kwargs)


@wraps(StdQOpONNXManager.export)
def export_standard_qop_onnx(*args, **kwargs):
    return StdQOpONNXManager.export(*args, **kwargs)


@wraps(StdQCDQONNXManager.export)
def export_standard_qcdq_onnx(*args, **kwargs):
    return StdQCDQONNXManager.export(*args, **kwargs)


@wraps(PytorchQuantManager.export)
def export_pytorch_quant(*args, **kwargs):
    return PytorchQuantManager.export(*args, **kwargs)