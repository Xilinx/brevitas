import torch

from brevitas.export.shark.manager import SharkManager
import brevitas.nn as qnn

n = qnn.QuantLinear(3, 5, bias=False)
n.eval()

export = SharkManager()

ds = export.export(n, torch.randn(1, 3))
print(ds)
ds.save('test_dataset.irpa')
