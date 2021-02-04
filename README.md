# Brevitas

[![Gitter](https://badges.gitter.im/xilinx-brevitas/community.svg)](https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
![Pytest](https://github.com/Xilinx/brevitas/workflows/Pytest/badge.svg?branch=master)
![Examples Pytest](https://github.com/Xilinx/brevitas/workflows/Examples%20Pytest/badge.svg?branch=master)
[![DOI](https://zenodo.org/badge/140494324.svg)](https://zenodo.org/badge/latestdoi/140494324)

Brevitas is a Pytorch library for quantization-aware training (QAT).

*Brevitas is currently under active development. Documentation, tests, examples, and pretrained models will be progressively released.*

**Please note that Brevitas is a research project and not an official Xilinx product.**

## History

*2021/02/04* - Release version 0.2.1, includes various bugfixes of QuantTensor w/ zero-point.
*2021/01/30* - First release version 0.2.0 on PyPI.

## Requirements

* Python >= 3.6.
* [Pytorch](https://pytorch.org) >= 1.1.0 (minimal), 1.3.1 (suggested).
* Windows, Linux or macOS.
* GPU training-time acceleration (*Optional* but recommended).

## Installation

##### Installing from PyPI

You can install the latest release from PyPI:
```bash
pip install brevitas
```

##### Installing from Github

To get the very latest version, you can install directly from GitHub:
```bash
pip install git+https://github.com/Xilinx/brevitas.git
```

## Introduction

Brevitas implements a set of building blocks at different levels of abstraction to model a reduced precision hardware data-path at training time. 

Brevitas provides a platform both for researchers interested in implementing new quantization-aware training techinques, as well as for practitioners interested in applying current techniques to their models.

The quantizers currently implemented support variations of uniform affine quantization. Non-uniform quantization is currently not supported.

## Getting started

Here's how a simple 4 bit weights, 8 bit activations LeNet looks like:


```python
from torch.nn import Module
import torch.nn.functional as F
from brevitas.nn import QuantIdentity, QuantConv2d, QuantReLU, QuantLinear

class QuantLeNet(Module):
    def __init__(self):
        super(QuantLeNet, self).__init__()
        self.quant_inp = QuantIdentity(bit_width=8)
        self.conv1 = QuantConv2d(3, 6, 5, weight_bit_width=4)
        self.relu1 = QuantReLU(bit_width=8)
        self.conv2 = QuantConv2d(6, 16, 5, weight_bit_width=4)
        self.relu2 = QuantReLU(bit_width=8)
        self.fc1   = QuantLinear(16*5*5, 120, bias=True, weight_bit_width=4)
        self.relu3 = QuantReLU(bit_width=8)
        self.fc2   = QuantLinear(120, 84, bias=True, weight_bit_width=4)
        self.relu4 = QuantReLU(bit_width=8)
        self.fc3   = QuantLinear(84, 10, bias=False, weight_bit_width=4)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
```

## Basic usage

Brevitas exposes a few built-in quantizers that can be found under `brevitas.quant`. A quantizer
can be assigned to a `brevitas.nn` layer to perform quantization of some part of it. 

For example, if we look at the source code of `QuantConv2d`, we can see that by default weights are 
quantized with the `Int8WeightPerTensorFloat`quantizer, which performs 8-bit signed integer 
quantization with a per-tensor floating-point scale factor, while input, output and bias quantization 
are disabled. For reference, this a type of weight quantization supported by the ONNX standard.

```python
from torch.nn import Conv2d
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat

class QuantConv2d(..., Conv2d):
    
    def __init__(
            self, 
            ...,
            weight_quant=Int8WeightPerTensorFloat, 
            bias_quant=None, 
            input_quant=None, 
            output_quant=None)
```

We can enable quantization of bias, input and output by setting appropriate quantizers, while keeping
the default weight quantizer:
```python
from brevitas.nn import QuantConv2d
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8Bias, Int8ActPerTensorFloat

conv = QuantConv2d(
    ...,
    bias_quant=Int8Bias, 
    input_quant=Int8ActPerTensorFloat, 
    output_quant=Int8ActPerTensorFloat)
```

Now let's say we want to set the bit-width of the weights to *4 bits* and enable *per-channel* scale factors. 
The simplest way is to pass appropriate keyword arguments that override the attributes defined in the 
default `Int8WeightPerTensorFloat` quantizer:

```python
from brevitas.nn import QuantConv2d
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8Bias, Int8ActPerTensorFloat

conv = QuantConv2d(
    ...,
    weight_bit_width=4,
    weight_scaling_per_output_channel=True,
    bias_quant=Int8Bias, 
    input_quant=Int8ActPerTensorFloat, 
    output_quant=Int8ActPerTensorFloat)
```

Any keyword argument with the prefix `weight_` is passed automatically to the quantizer assigned
to `weight_quant`, and possibly overrides any pre-existing attribute with the same name defined there. 
The same principle applies to `bias_`, `input_` and `output_`.

When the same arguments are re-used across various layers, it can also be convenient to simply 
define a new quantizer instead. One way to do so is by inheriting from an existing quantizer:

```python
from brevitas.nn import QuantConv2d
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.quant.scaled_int import Int8Bias, Int8ActPerTensorFloat

class MyWeightQuant(Int8WeightPerTensorFloat):
    bit_width = 4
    scaling_per_output_channel = True

conv1 = QuantConv2d(..., weight_quant=MyWeightQuant)
conv2 = QuantConv2d(..., weight_quant=MyWeightQuant)
```

Given a quantizer, it can be difficult at first to see which options are available to modify. The best way
is to look at how the quantizer is defined. In the case of `Int8WeightPerTensorFloat`,
we see that it simply inherits from `NarrowIntQuant`, `MaxStatsScaling`, `PerTensorFloatScaling8bit`:

```python
from brevitas.quant.base import NarrowIntQuant, MaxStatsScaling, PerTensorFloatScaling8bit

class Int8WeightPerTensorFloat(
    NarrowIntQuant, MaxStatsScaling, PerTensorFloatScaling8bit):
    pass
``` 

What that means is that the quantizer is the composition of those three complementary parts, which
are defined in `brevitas.quant.base` as follows: 

```python
from brevitas.inject import BaseInjector as Injector
from brevitas.inject.enum import QuantType, BitWidthImplType, ScalingImplType
from brevitas.inject.enum import RestrictValueType, StatsOp
from brevitas.core.zero_point import ZeroZeroPoint

class NarrowIntQuant(Injector):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.CONST
    narrow_range = True
    signed = True
    zero_point_impl = ZeroZeroPoint

class MaxStatsScaling(Injector):
    scaling_impl_type = ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX

class PerTensorFloatScaling8bit(Injector):
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.FP
    bit_width = 8
``` 

We can see that various attributes of the quantizer come from *enums*. Within Brevitas, enums
provide an abstraction that hides away various implementation details, while exposing only a simplified 
interface that can be easily explored through auto-complete or source-code inspection.

For example, let's say that we want to implement a quantizer where the bit-width is a parameter
learned with backprop starting from an initialization of 4-bits, while the scale factor is also a parameter,
but it's a floating-point number learned in logarithmic domain,
initialized based on the maximum value found in the tensor to quantize. It's just a matter
of switching a few enums:

```python
from brevitas.inject import BaseInjector as Injector
from brevitas.inject.enum import QuantType, BitWidthImplType, ScalingImplType
from brevitas.inject.enum import RestrictValueType, StatsOp
from brevitas.core.zero_point import ZeroZeroPoint
from brevitas.nn import QuantConv2d

class MyLearnedWeightQuant(Injector):
    quant_type = QuantType.INT
    bit_width_impl_type = BitWidthImplType.PARAMETER  # previously was BitWidthImplType.CONST
    narrow_range = True
    signed = True
    zero_point_impl = ZeroZeroPoint
    scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS  # previously was ScalingImplType.STATS
    scaling_stats_op = StatsOp.MAX
    scaling_per_output_channel = False
    restrict_scaling_type = RestrictValueType.LOG_FP  # previously was RestrictValueType.FP
    bit_width = 4

conv = QuantConv2d(..., weight_quant=MyLearnedWeightQuant)
``` 

In Brevitas enums are always equal to their string representation in a case-insensitive way.
So to go back to the initial example of overriding attributes of a quantizer with keyword arguments,
we could have also wrote `MyLearnedWeightQuant` implicitly by simply overriding a few attributes of the default 
weight quantizer:

```python
from brevitas.nn import QuantConv2d

conv = QuantConv2d(
    ..., 
    weight_bit_width=4,
    weight_bit_width_impl_type='parameter', 
    weight_scaling_impl_type='parameter_from_stats',
    weight_restrict_scaling_type='log_fp')
```

Everything that has been said so far applies to quantized activations too, with just a couple of variations.
For reasons of retro-compatibility with older versions of Brevitas, the output quantizer of an activation layer
is called `act_quant`, and passing keyword arguments to it doesn't require a prefix. 
So for example, to set the output of a `QuantReLU` to 4-bits, it's enough to write:

```python
from brevitas.nn import QuantReLU

act = QuantReLU(bit_width=4)
```


## Settings

Brevitas exposes a few settings that can be toggled through env variables.

- **BREVITAS_JIT=1** (*Default: = 0*): Enables compilation of the available built-in quantizers through TorchScript just-in-time compiler, 
  together with a small native .cpp extension for the straight-through estimator functions. This can provide a speed-up and/or memory savings at training time. 
  Please note that under certain circumstances this has been shown to produce diverging results compared to BREVITAS_JIT=0. Use at your own risk. 

- **BREVITAS_VERBOSE=1** (*Default: = 0*): Enables verbose compilation of the straight-through estimator functions native extension.

- **BREVITAS_IGNORE_MISSING_KEYS=1** (*Default: =0*): Ignore errors related to missing *state_dict* values when loading a pre-trained model on top of a Brevitas model.
    This is typically enabled when re-training from a floating-point checkpoint.

## F.A.Q.

**Q: How can I train X/Y and run it on hardware W/Z? I can't find any documentation.**

**A:** Brevitas is still sparsely documented. Until the situation improves, feel free to open an issue or ask on our gitter channel.


**Q: Training with Brevitas is slow and/or I can't fit the same batch size as with floating-point training. Why? What can I do?**

**A:** Quantization-aware training involves a lot of element-wise operations, 
which carry low arithmetic intensity and contribute to a more involved computational graph during backpropragation. 
As such, it typically ends up being slower and more resource-intensive than standard floating-point training. 

Brevitas in particular is biased towards greater flexibility, at the cost of some training-time effieciency. 
The general principle is that it's trading off more complexity at training time for more efficiency at inference time.

To mitigate somewhat the slow-down, try enabling *BREVITAS_JIT* as reported in the *Settings* section.


**Q: Inference with Brevitas is slow. I thought the point of QAT was to make my model faster at inference time. What I am doing wrong?**

**A:** Brevitas is concerned with modelling a reduced precision data-path, it does not provide inference-time acceleration on its own. 
To achieve acceleration, you should export your Brevitas model to a downstream toolchain / backend. 

Brevitas can currently export to:
- FINN  - for dataflow acceleration on Xilinx FPGAs. 
- PyXIR (*experimental*) - for DPU acceleration on Xilinx FPGAs. 
- Standard ONNX (*experimental*) - for acceleration with e.g. onnxruntime, or any other ONNX-compliant toolchain.
- Pytorch's *quantized.functional* operators (*experimental*) - for acceleration through Pytorch itself,
  or any additional downstream toolchains supported by Pytorch (e.g. TVM).

Because Brevitas implements a super-set of layers and datatypes supported by various downstream toolchains and hardware platforms, 
the result is that each export flow supports only a certain subset of features, in ways that are not necessarely obvious. 
More examples and documentation will be released to illustrate the various restrictions imposed by each target platform.
As a general note though, currently FINN is the only toolchain that supports export of operators quantized to below 8-bit.

**Q: My (C/G/T)PU supports float16 / bfloat16 / bfloat19 training. Can I use it to train with Brevitas?**

**A:** Datatypes outside of float32 at training time have not been tested. That includes training on TPU / Pytorch-XLA.
Do the math in terms of which reduced-precision integers can reasonably fit in a reduced-precision 
floating-point format at training time, and use at your own risk.


## Author

Alessandro Pappalardo (@volcacius) @ Xilinx Research Labs.


