import torch

from brevitas_examples.common.svd_quant import LayerwiseLowRankCorrection


@torch.no_grad()
def apply_svd_quant(model, blacklist=None, rank=32, iters=1, dtype=torch.float32):
    return LayerwiseLowRankCorrection(model, blacklist).apply(
        rank=rank, iters=iters, dtype=dtype)[0]
