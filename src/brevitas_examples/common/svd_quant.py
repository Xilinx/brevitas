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
        x_device = x.device
        l_device = self.layer.weight.device
        self.correction = self.correction.to(x_device)
        self.layer = self.layer.to(x_device)
        out = self.layer(x) + self.correction(x)
        self.correction = self.correction.to(l_device)
        self.layer = self.layer.to(l_device)
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


def _create_correction_module(layer, rank, iters=1, dtype=torch.float32):

    def calculate_Err(orig_weight, L1, L2, layer):
        R = orig_weight - (layer.quant_weight() + L2 @ L1)
        Err = torch.norm(R) / torch.norm(orig_weight)
        return Err

    in_features = layer.weight.shape[1]
    out_features = layer.weight.shape[0]
    source_dtype = layer.weight.dtype
    source_device = layer.weight.device

    # Convert to dtype for SVD
    # For some reason not fully clear to me, if we don't put hits on CPU then there's memory leakage
    orig_weight = layer.weight.detach().to(dtype=dtype).cpu()
    cm = LowRankCorrectionModule(in_features, out_features, rank).to(
        dtype=dtype, device=layer.weight.device)
    next_R = orig_weight
    best_R = orig_weight
    best_L1 = cm.l1.weight.detach()
    best_L2 = cm.l2.weight.detach()
    best_Err = calculate_Err(orig_weight.to(source_device), best_L1, best_L2, layer)
    logging.debug(f"Start Residual: {best_Err}")
    for i in range(iters):
        U, S, V = torch.linalg.svd(next_R)
        L1 = torch.diag(S[:rank]) @ V[:rank, :]
        L2 = U[:, :rank]
        R = orig_weight - L2 @ L1
        layer.weight = torch.nn.Parameter(R)

        next_R = orig_weight - layer.quant_weight()
        cur_Err = calculate_Err(orig_weight, L1, L2, layer)
        if cur_Err < best_Err:
            logging.debug(f"Best residual at iteration {i}: {cur_Err}")
            best_Err = cur_Err.cpu()
            best_L1 = L1.cpu()
            best_L2 = L2.cpu()
            best_R = R.cpu()
        del U, S, V, L1, L2, R
    cm.l1.weight = torch.nn.Parameter(best_L1)
    cm.l2.weight = torch.nn.Parameter(best_L2)
    layer.weight = torch.nn.Parameter(best_R)

    layer = layer.cpu()
    cm = cm.cpu()
    ecm = ErrorCorrectedModule(cm, layer)
    ecm.to(dtype=source_dtype)
    del best_L1, best_L2, best_R, next_R, orig_weight
    return ecm, best_Err


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

    def apply(self, rank, iters=1, dtype=torch.float32):
        model = self.model
        errs = torch.zeros((len(self.layers),))
        rewriters = []
        for i, layer in enumerate(tqdm(self.layers)):
            ecm, err = _create_correction_module(layer, rank, iters=iters, dtype=dtype)
            errs[i] = err.to(dtype=errs.dtype, device='cpu')
            rewriters.append(ModuleInstanceToModuleInstance(layer, ecm))
            torch.cuda.empty_cache()
        for r in rewriters:
            model = r.apply(model)
        return model, errs
