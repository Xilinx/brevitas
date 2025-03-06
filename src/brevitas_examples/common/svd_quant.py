from typing import List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from brevitas.graph.base import GraphTransform
from brevitas.graph.base import ModuleInstanceToModuleInstance
import brevitas.nn as qnn
from brevitas.utils.logging import setup_logger

_supported_layers = (qnn.QuantLinear,)
logging = setup_logger(__name__)


class ErrorCorrectedModule(torch.nn.Module):

    def __init__(self, correction_module: torch.nn.Module, layer: torch.nn.Module) -> None:
        super().__init__()
        self.correction = correction_module
        self.layer = layer

    def forward(self, x):
        self.correction = self.correction.to('cuda')
        self.layer = self.layer.to('cuda')
        out = self.layer(x) + self.correction(x)
        self.correction = self.correction.to('cpu')
        self.layer = self.layer.to('cpu')
        return out


class LowRankCorrectionModule(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(in_features, rank, bias=False)
        self.l2 = torch.nn.Linear(rank, out_features, bias=False)
        self.l1.weight.data = torch.zeros_like(self.l1.weight.data)
        self.l2.weight.data = torch.zeros_like(self.l2.weight.data)

    def forward(self, x):
        return self.l2(self.l1(x))


def _create_correction_module(layer, rank):
    in_features = layer.weight.shape[1]
    out_features = layer.weight.shape[0]
    source_dtype = layer.weight.dtype

    # Convert to FP32 for SVD
    U, S, V = torch.linalg.svd(layer.weight.to(dtype=torch.float32))
    L1 = torch.diag(S[:rank]) @ V[:rank, :]
    L2 = U[:, :rank]
    R = layer.weight - L2 @ L1
    logging.debug(f"Est. Variance Retained: {S[:rank].sum() / S.sum()}")
    logging.debug(f"Residual: {torch.norm(R) / torch.norm(layer.weight)}")
    cm = LowRankCorrectionModule(in_features, out_features, rank)
    cm.l1.weight = torch.nn.Parameter(L1)
    cm.l2.weight = torch.nn.Parameter(L2)
    layer.weight = torch.nn.Parameter(R)

    # if layer.weight_quant.is_quant_enabled:
    #     layer.weight_quant.init_tensor_quant()
    ecm = ErrorCorrectedModule(cm, layer).to(dtype=source_dtype)
    var = S[:rank].sum() / S.sum()
    del U, S, V, L1, L2, R
    return ecm, var


class LayerwiseLowRankCorrection(GraphTransform):

    def __init__(self, model, blacklist_layers: Optional[List[str]] = None):
        self.model = model
        self.blacklist_layers = blacklist_layers
        self.layers = []
        self.find_module(model, self.layers)

    def find_module(self, model, layers: List, prefix=''):
        """
        Iterate through the model looking at immediate children of every module to look for supported modules.
        This allows us to stop the search when we meet a top-level module that is supported.
        """
        if isinstance(model, _supported_layers):
            if self.blacklist_layers is not None and prefix in self.blacklist_layers:
                return
            layers += [model]
        else:
            for name, module in model.named_children():
                full_name = prefix + '.' + name if prefix != '' else name
                self.find_module(module, layers, full_name)

    def apply(self, rank):
        model = self.model
        variances = torch.zeros((len(self.layers),))
        rewriters = []
        for i, layer in enumerate(tqdm(self.layers)):
            ecm, var = _create_correction_module(layer, rank)
            variances[i] = var.to(dtype=variances.dtype)
            rewriters.append(ModuleInstanceToModuleInstance(layer, ecm).apply(model))
            ecm.cpu()
            torch.cuda.empty_cache()

        for r in rewriters:
            model = r.apply(model)
        return model, variances
