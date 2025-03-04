import torch

from brevitas_examples.common.svd_quant import LayerwiseLowRankCorrection


@torch.no_grad()
def apply_svd_quant(model, blacklist=None, rank=32):
    return LayerwiseLowRankCorrection(model, blacklist).apply(rank)[0]
