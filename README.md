# Brevitas

[![Gitter](https://badges.gitter.im/xilinx-brevitas/community.svg)](https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
![Pytest](https://github.com/Xilinx/brevitas/workflows/Pytest/badge.svg?branch=master)
![Examples Pytest](https://github.com/Xilinx/brevitas/workflows/Examples%20Pytest/badge.svg?branch=master)
[![DOI](https://zenodo.org/badge/140494324.svg)](https://zenodo.org/badge/latestdoi/140494324)

Brevitas is a Pytorch library for quantization-aware training.

*Brevitas is currently under active development and on a rolling release. It should be considered in beta stage. Minor API changes are still planned. Documentation, tests, examples, and pretrained models will be progressively released.*

## Requirements

* Python >= 3.6
* [Pytorch](https://pytorch.org) >= 1.1.0 (minimal), 1.3.1 (suggested)


## Installation

##### Installing from master

You can install the latest master directly from GitHub:
```bash
pip install git+https://github.com/Xilinx/brevitas.git
```

## Introduction

Brevitas implements a set of building blocks at different levels of abstraction to model a reduced precision hardware data-path at training time. 

Brevitas provides a platform both for researchers interested in implementing new quantization-aware training techinques, as well as for practitioners interested in applying current techniques to their models.

## Getting started

Here's how a simple 4 bit weights, 8 bit activations LeNet looks like, using default settings for scaling:


```python
from torch.nn import Module
import torch.nn.functional as F
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU
from brevitas.core.quant import QuantType

class QuantLeNet(Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.quant_inp = QuantIdentity(bit_width=8, min_val=-1.0, max_val=1.0)
        self.conv1 = QuantConv2d(3, 6, 5, weight_bit_width=4)
        self.relu1 = QuantReLU(bit_width=8, max_val=6)
        self.conv2 = QuantConv2d(6, 16, 5, weight_bit_width=4)
        self.relu2 = QuantReLU(bit_width=4, max_val=6)
        self.fc1   = QuantLinear(16*5*5, 120, bias=True, weight_bit_width=4)
        self.relu3 = QuantReLU(bit_width=8, max_val=6)
        self.fc2   = QuantLinear(120, 84, bias=True, weight_bit_width=4)
        self.relu4 = QuantReLU(bit_width=8, max_val=6)
        self.fc3   = QuantLinear(84, 10, bias=False, weight_bit_width=4)

    def forward(self, x):
        out = self.inp(x)
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
```

## Author

Alessandro Pappalardo (@volcacius) @ Xilinx Research Labs.


