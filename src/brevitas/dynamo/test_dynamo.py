from typing import List
import time

import torch
import torch._dynamo as dynamo
from torchvision.models import resnet18

from brevitas.dynamo.compile import quantized_ort

CALIB_ITERS = 10
TOTAL_ITERS = 100

@dynamo.optimize(quantized_ort(CALIB_ITERS))
def run_model(model, *args, **kwargs):
    return model(*args, **kwargs)

model = resnet18(pretrained=True)
model.eval()
for i in range(TOTAL_ITERS):
    print(f"iter {i}")
    start = time.time()
    output = run_model(model, torch.randn(1, 3, 224, 224))
    print(f"Took {time.time() - start} ms")
