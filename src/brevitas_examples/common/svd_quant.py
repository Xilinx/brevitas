from typing import List, Optional

import torch
import torch.nn as nn

from brevitas.graph.base import GraphTransform
from brevitas.graph.base import ModuleInstanceToModuleInstance
import brevitas.nn as qnn

_supported_layers = (qnn.QuantLinear,)


class ErrorCorrectedModule(torch.nn.Module):

    def __init__(self, correction_module: torch.nn.Module, layer: torch.nn.Module) -> None:
        super().__init__()
        self.correction = correction_module
        self.layer = layer

    def forward(self, x):
        return self.layer(x) + self.correction(x)


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
    train = layer.train
    in_features = layer.weight.shape[1]
    out_features = layer.weight.shape[0]
    source_dtype = layer.weight.dtype

    # Convert to FP32 for SVD
    U, S, V = torch.linalg.svd(layer.weight.to(dtype=torch.float32))
    L1 = torch.diag(S[:rank]) @ V[:rank, :]
    L2 = U[:, :rank]
    R = layer.weight - L2 @ L1
    print(f"Est. Variance Retained: {S[:rank].sum() / S.sum()}")
    print(f"Residual: {torch.norm(R) / torch.norm(layer.weight)}")
    cm = LowRankCorrectionModule(in_features, out_features, rank)
    cm.l1.weight.data = L1
    cm.l2.weight.data = L2
    layer.weight.data = R
    ecm = ErrorCorrectedModule(cm, layer)
    ecm.train = train
    ecm.to(dtype=source_dtype)
    return ecm, S[:rank].sum() / S.sum()


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

    def setup(self):
        pass

    def apply(self, rank):
        variances = torch.zeros((len(self.layers),))
        for i, layer in enumerate(self.layers):
            ecm, var = _create_correction_module(layer, rank)
            variances[i] = var.to(dtype=variances.dtype)
            rewriter = ModuleInstanceToModuleInstance(layer, ecm)
            rewriter.apply(self.model)
        return variances
