from brevitas.nn import QuantLSTMCELL
import torch.nn as nn
from LSTMcell import *
import torch

if __name__ == '__main__':
    weight_config = {}
    # weight_config['quant_type'] = 'QuantType.INT'
    activation_config = {}
    state1 = torch.rand(2,500)
    state2 = torch.rand(2,500)
    state = (state1,state2)
    A = QuantLSTMCELL(100, 500, weight_config, activation_config)
    B = LSTMCell(100,500)

    input = torch.rand(2, 100)

    C = A(input,state)
    D = B(input, state)
    print('END')
    print(A.graph_for(input,state))
    print(B.graph_for(input,state))