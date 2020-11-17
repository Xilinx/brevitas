from torch.nn import Module

from brevitas.export.base import BaseManager, _set_export_mode
from .handler import PytorchQuantConv2dHandler
from .handler import PytorchQuantConv1dHandler
from .handler import PytorchQuantLinearHandler


class PytorchQuantManager(BaseManager):

    handlers = [
        PytorchQuantConv1dHandler,
        PytorchQuantConv2dHandler,
        PytorchQuantLinearHandler]

    @classmethod
    def export(
            cls,
            module: Module):
        module = module.eval()
        module.apply(cls.set_export_handler)
        module.apply(lambda m: _set_export_mode(m, enabled=True))
        return module