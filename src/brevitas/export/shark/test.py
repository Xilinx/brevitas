from sharktank.layers import LinearLayer
from sharktank.types import Theta
import torch

from brevitas.export.shark.manager import SharkManager
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant.experimental.float import Fp8e4m3ActPerTensorFloat
from brevitas.quant.experimental.float_quant_fnuz import Fp8e4m3FNUZActPerTensorFloat

inp = torch.randn(1, 3)
n = qnn.QuantLinear(3, 5, input_quant=Fp8e4m3FNUZActPerTensorFloat, bias=False)
n.eval()
o = n(inp)

export = SharkManager()

ds = export.export(n, torch.randn(1, 3))
ds.save('test_dataset.irpa')
theta = ds.root_theta
linear = LinearLayer(theta, fake_quant=False)
actual = linear(inp)
actual = actual.to(dtype=o.dtype).view(o.shape)

abs_diff = (o - actual).abs().max()
torch.testing.assert_close(actual, o)
