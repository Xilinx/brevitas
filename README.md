# Brevitas

[![DOI](https://zenodo.org/badge/140494324.svg)](https://zenodo.org/badge/latestdoi/140494324)

Brevitas is a Pytorch library for training-aware quantization.

*Brevitas is currently under active development and to be considered in alpha stage. APIs might and probably will change. Documentation, examples, and pretrained models will be progressively released.*

## Requirements
* [Pytorch](https://pytorch.org) >= 1.1.0

## Introduction

Brevitas implements a set of building blocks to model a reduced precision hardware data-path at training time.
While partially biased towards modelling dataflow-style, very low-precision implementations, the building blocks can be parametrized and assembled together to target all sorts of reduced precision hardware.

The implementations tries to adhere to the following design principles:
- Idiomatic Pytorch, when possible.
- Modularity first, at the cost of some verbosity.
- Easily extendible.

## Target audience
Brevitas is mainly targeted at researchers and practicioners in the fields of training for reduced precision inference. 

The implementation is quite rich in options and allows for very fine grained control over the trained model. 
However, compared to other software solutions in this space, the burden of correctly modelling the target data-path is currently placed on the user. 


## Installation

##### Installing from master

You can install the latest master directly from GitHub:
```bash
pip install git+https://github.com/Xilinx/brevitas.git
```

##### Dev install

Alternatively, you can install a dev copy straight from the cloned repo:

```bash
git clone https://github.com/Xilinx/brevitas.git
cd brevitas
pip install -e .
```

## Features

Brevitas' features are organized along the following (mostly) orthogonal axes:

- **Quantization type**: binary, ternary, or uniform integer quantization.
- **Scaling**: support for various shapes, learning strategies and constraints.
- **Precision**: constant or learned bit-width.
- **Target tensor**: weights, activations or accumulators.
- **Cost**: model the hardware cost at training-time.


## Supported Layers
The following layers and operations are supported out-of-the-box:

- QuantLinear
- QuantConv2d
- QuantReLU, QuantHardTanh, QuantTanh, QuantSigmoid
- QuantAvgPool2d
- QuantBatchNorm2d
- Element-wise add, concat
- Saturating integer accumulator

Additional layers can be easily supported using a combination of pre-existing modules.

## Getting started

Here's how a simple quantized LeNet might look like, starting from a ReLU6 for activations and using default settings for scaling:


```python
from torch.nn import Module
import torch.nn.functional as F
import brevitas.nn as qnn
from brevitas.core.quant import QuantType

class QuantLeNet(Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, 
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8)
        self.relu1 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.conv2 = qnn.QuantConv2d(6, 16, 5, 
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8)
        self.relu2 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc1   = qnn.QuantLinear(16*5*5, 120, bias=True, 
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8)
        self.relu3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc2   = qnn.QuantLinear(120, 84, bias=True, 
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8)
        self.relu4 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc3   = qnn.QuantLinear(84, 10, bias=False, 
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8)

    def forward(self, x):
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

## Code organization

The implementation in organized in three different levels, each corresponding to a package:
- **core**: contains the implementation of all Brevitas' quantization routines, leveraging the *JIT* as much as possible.
- **proxy**: combines the routines implemented in the core in way that makes sense for different target tensors, namely weights, activations and accumulators.
- **nn**: wraps the different proxies in a user-facing API, which can be used as a drop-in replacement for torch.nn modules. 

Additionally, the following packages are present:
- **function**: implements various support routines.
- **cost**: exposes different loss functions for modelling the hardware cost at training-time or for regularization purposes. 

## A note on the implementation

Brevitas operates (possibly but not strictly) on a *QuantTensor*, currently implemented as a *NamedTuple* because of a lack of support for custom Tensor sub-classes in Pytorch. This might change in the future.

A QuantTensor propagates the following information:
- **quant_tensor**: the quantized tensor, in dequantized representation (i.e. floating-point order of magnitude).
- **scale_factor**: the scale factor implicit in *quant_tensor*.
- **bit_width**: the precision of *quant_tensor* in bits. 

Propagating scale factors and bit-width along the forward pass allows to model and operate on accumulators, whose bit-width and scale factor depends on the layers contributing to them. 
However, propagating a tuple of values in a forward() call breaks compatibility with certain modules like *nn.Sequential*, which assumes 1 input and 1 output. As such, operating on a QuantTensor is optional.


## Author

Alessandro Pappalardo @ Xilinx Research Labs.
