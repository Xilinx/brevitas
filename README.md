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
- **Target tensor**: weights, activations or accumulators.
- **Scaling**: support for various shapes, learning strategies and constraints.
- **Precision**: constant or learned bit-width.
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

## Pretrained models

Below a list of relevant pretrained models available, currently for evaluation only. Look at the `examples` folder for how to verify accuracy. Training scripts will be made available in the near future.


|   | Name         | Scaling Type               | First layer weights | Weights | Activations | Avg pool | Top1  | Top5  | Pretrained model                                                                                | Retrained from                                                |
|---|--------------|----------------------------|---------------------|---------|-------------|----------|-------|-------|-------------------------------------------------------------------------------------------------|---------------------------------------------------------------|
|   | MobileNet V1 | Floating-point per channel | 8 bit               | 4 bit   | 4 bit       | 4 bit    | 71.25 | 90.10 | [Download](https://github.com/Xilinx/brevitas/releases/download/examples-0.0.1/mobilenetv1.pth) | [link](https://github.com/osmr/imgclsmob/tree/master/pytorch) |


## Scaling

Brevitas supports multiple alternative options for scaling factors.


With respect to the number of dimensions:
- Per tensor or per (output) channel scaling, for both weights and activations.
For activations, per channel scaling usually makes sense only before a depth-wise separable convolution.

With respect to how they are learned:
- As a standalone Pytorch *Parameter*, initialized from statistics (for weights) or from user-defined values (for activations).
- as a statistical function of the full-precision version of the target tensor, possibly along with affine coefficients. 
  Various function as supported, such as *max(abs(x))*, and more can be easily added. 
  For activations, learning of this kind is done in the style of batch-norm, i.e. statistics are collected at training-time to use them at inference time.
  In all cases, gradients are allowed to backprop through the defined function for best accuracy results.
  
Possibly shared between different layers that have to scaled equally:
- By sharing the underlying Parameter, when the scale factor is a learned parameter.
- By applying the statistical function to a concatenation of the different set of weights involved, when the scale factors are learned as a function of the weights.

Possibly constrained to:
- A standard floating point number, i.e. no restrictions.
- A floating point power-of-two exponent, i.e. a floating point number learned in log domain.
- An integer power-of-two value exponent, i.e. rounding the above fp log version to the next integer.


## Precision
Brevitas supports both constant and learned precision.

In an quantization flow leveraging integer uniform quantization, the *bit-width* (together with the *sign*) determines the *min* and *max* integer values
used for scaling and clamping. Assuming that an integer bit-width can be modelled as the rounded copy of an underlying floating point value (with a straight-through estimator in the backward pass), 
all the operations involving bit-width are differentiable. As such, the bit-width can a learned parameter, 
without resorting to more complicated approaches leveraging AutoML or Gumbel-Softmax categorical variables.

- For weights and activation:
Learned bit-width is equal to *base_bit_width + round_ste(abs(offset))*, where *base_bit_width* is a constant representing the minimum bit-width to converge to (required to be *>= 2*) and offset is a learned parameter.

- For modelling an accumulator saturate to a learned bit width:
The bit-width of the accumulator (computed by upstream layers) is taken as an input, and a learned negative offset is subtracted from it.
In order to avoid conflicting with regularization losses that promotes small magnitude of learned parameters, such as weight-decay, 
the offset is implemented with the learned parameter at the denominator, so that smaller values results in reduced overall bit-width.

Additional requirements or relaxations can be put on the resulting bit-width:
- Avoid the rounding step, and learn a floating point bit-width first (with the goal of retrained afterwards).
- Constrain the bit width to power-of-two values, e.g. 2, 4, 8.
- Share the learned bit-width between two or more layers, in order to e.g. keep the precision of a Conv2d layer and the precision of its input the same.



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
