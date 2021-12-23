from brevitas.nn.quant_rnn import QuantRNNLayer
import torch

m = QuantRNNLayer(10, 15)
m(torch.randn(2, 10, 10))