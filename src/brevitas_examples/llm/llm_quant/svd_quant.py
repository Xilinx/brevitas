
import torch

@torch.no_grad()
def apply_svd_quant(model, blacklist=None, rank=32):
    LayerwiseLowRankCorrection(model, blacklist).apply(rank)
