# Brevitas

[![Gitter](https://badges.gitter.im/xilinx-brevitas/community.svg)](https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
![Pytest](https://github.com/Xilinx/brevitas/workflows/Pytest/badge.svg?branch=master)
![Examples Pytest](https://github.com/Xilinx/brevitas/workflows/Examples%20Pytest/badge.svg?branch=master)
[![DOI](https://zenodo.org/badge/140494324.svg)](https://zenodo.org/badge/latestdoi/140494324)

Brevitas is a PyTorch research library for quantization-aware training (QAT).

*Brevitas is currently under active development. Documentation, examples, and pretrained models will be progressively released.*

**Please note that Brevitas is a research project and not an official Xilinx product.**

If you like this project please consider ⭐ this repo, as it is the simplest and best way to support it. 

If you have issues, comments, or are just looking for advices on training quantized neural networks, you can open an issue, a discussion, or chat over in our [gitter](https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) channel.


## History
- *2021/03/04* - Release version 0.3.1, fix bug w/ act initialization from statistics w/ IGNORE_MISSING_KEYS=1.
- *2021/03/01* - Release version 0.3.0, implements enum and shape solvers within extended dependency injectors. This allows declarative quantizers to be self-contained.
- *2021/02/04* - Release version 0.2.1, includes various bugfixes of QuantTensor w/ zero-point.
- *2021/01/30* - First release version 0.2.0 on PyPI.

## Requirements

* Python >= 3.6.
* [Pytorch](https://pytorch.org) >= 1.1.0 (minimal), >= 1.5.0 (suggested).
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
It provides a platform both for researchers interested in implementing new quantization-aware training techinques, as well as for practitioners interested in applying current techniques to their models, with the aim of bridging the gap between research and the industry around quantization.

Brevitas supports a super-set of quantization schemes implemented across various frameworks and compilers under a single unified API.  
For certain combinations of layers and types of of quantization inference acceleration is supported by exporting to *FINN*, *onnxruntime*, *Pytorch*'s own quantized inference operators, *TVM* (through the Pytorch export flow), and *PyXIR*.

Brevitas has been successfully adopted both in various research projects as well as in large-scale commercial deployments targeting custom accelerators running on Xilinx FPGAs. The general quantization style implemented is affine quantization, with a focus on uniform quantization. Non-uniform quantization is currently not supported out-of-the-box.


## Getting started

### Low-precision LeNet for integer-only FPGA acceleration with FINN 

Here's how a simple 3 bit weights, 4 bit activations LeNet for CIFAR10 looks like:


```python
from torch.nn import Module
import torch.nn.functional as F

import brevitas.nn as qnn


class LowPrecisionLeNet(Module):
    def __init__(self):
        super(LowPrecisionLeNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(
            bit_width=4, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(
            3, 6, 5, weight_bit_width=3, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(
            6, 16, 5, weight_bit_width=3, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc1   = qnn.QuantLinear(
            16*5*5, 120, bias=True, weight_bit_width=3, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc2   = qnn.QuantLinear(
            120, 84, bias=True, weight_bit_width=3, return_quant_tensor=True)
        self.relu4 = qnn.QuantReLU(
            bit_width=4, return_quant_tensor=True)
        self.fc3   = qnn.QuantLinear(
            84, 10, bias=False, weight_bit_width=3)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
```

Note a couple of things:
- We propagate `QuantTensor` between quantized layers by setting return_quant_tensor=True. Although not strictly necessary in this scenario, in helps in aiding debugging by making quantization attributes along the forward path explicit. 
- We quantize activations *after* non-linearities by instantiating multiple `QuantReLU` with `bit_width=4`. By default `QuantReLU` is *stateful* (scale is a learned parameter), so there is a difference between instantiating one `QuantReLU` that is called multiple times, and instantiating multiple `QuantReLU` that are each called once. 

The network defined above can be mapped to a low-precision *integer-only* dataflow accelerator implemented on a Xilinx FPGA by exporting it to FINN through a custom ONNX-based representation: 

```python
from brevitas.export import FINNManager

low_precision_lenet = LowPrecisionLeNet()

# ... training ...

FINNManager.export(low_precision_lenet, input_shape=(1, 1, 32, 32), export_path='finn_lenet.onnx')
```

### A mixed float-integer LeNet

Targeting other inference frameworks that support a mixture of floating-point and quantized layers, like onnxruntime and PyTorch itself, is also supported.
In this case, we mark whether a quantized layer should keep its output quantized or not with `return_quant_tensor`.
Any layer with `return_quant_tensor=False` (which is default) dequantizes its output to floating-point:  

```python
from torch import nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat as SignedWeightQuant
from brevitas.quant import ShiftedUint8WeightPerTensorFloat as UnsignedWeightQuant
from brevitas.quant import ShiftedUint8ActPerTensorFloat as ActQuant
from brevitas.quant import Int8Bias as BiasQuant


class ReducedRangeActQuant(ActQuant):
    bit_width = 7


class MixedFloatQuantLeNet(nn.Module):
    def __init__(self, bias_quant=True, reduced_act_quant=False, weight_signed=False):
        super(MixedFloatQuantLeNet, self).__init__()
        
        bias_quant   = BiasQuant if bias_quant else None
        act_quant    = ReducedRangeActQuant if reduced_act_quant else ActQuant
        weight_quant = SignedWeightQuant if weight_signed else UnsignedWeightQuant
        
        self.conv1 = qnn.QuantConv2d(
            1, 6, 5, input_quant=act_quant, weight_quant=weight_quant,
            output_quant=act_quant, bias_quant=bias_quant, return_quant_tensor=True)
        self.relu1 = nn.ReLU()
        self.conv2 = qnn.QuantConv2d(
            6, 16, 5, weight_quant=weight_quant, output_quant=act_quant,
            bias_quant=bias_quant, return_quant_tensor=True)
        self.relu2 = nn.ReLU()
        self.fc1   = qnn.QuantLinear(
            256, 120, bias=True, weight_quant=weight_quant,
            bias_quant=bias_quant, output_quant=act_quant)
        self.relu3 = nn.ReLU()
        self.fc2   = nn.Linear(120, 84, bias=True)
        self.relu4 = nn.ReLU()
        self.fc3   = nn.Linear(84, 10, bias=False)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out


```

Compared to the previous case, there are a few differences: 
- In this scenario activations are quantized before non-linearities by setting output quantizers on layers like `QuantConv2d` and `QuantLinear`. 
  This matches how frameworks like onnxruntime typically work.  
- Operations like *max-pooling* and *relu* that are invariant to quantization are implemented with standard PyTorch operators like `torch.nn.functional.max_pool2d` and `nn.ReLU()` even in the quantized portion of the network. 
  This requires at PyTorch 1.5.0, where support for `__torch_function__` was introduced. Users with older versions of PyTorch can replace them with (respectively) `QuantMaxPool2d(..., return_quant_tensor=True)` and `QuantReLU(act_quant=None, return_quant_tensor=True)`.  
- While in the previous example the default - scaled integer - quantized was being adopted and only a bit-width was specified, here the network is explicitly parametrized by quantizers taken from `brevitas.quant`. Again this is to match the kind of quantization supported by the frameworks shown below.

### Export to standard ONNX

After training, the above network can then be exported to an ONNX representation that complies with the standard opset:

```python
from brevitas.export import StdONNXManager

onnx_lenet = MixedFloatQuantLeNet()

# ... training ...

StdONNXManager.export(onnx_lenet, input_shape=(1, 1, 32, 32), export_path='onnx_lenet.onnx')
```

### Acceleration with onnxruntime

The generated output model can then be accelerated through any ONNX-compliant inference framework, such as *onnxruntime*:

```python
import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession('onnx_lenet.onnx')
input_name = sess.get_inputs()[0].name
pred_onx = sess.run(None, {input_name: np.random.randn(1, 1, 32, 32)})[0]
```

### Export to PyTorch quantized inference ops

With the same network definition it's also possible to target PyTorch's own quantized inference operators:

```python
from brevitas.export import PytorchQuantManager

pt_lenet = MixedFloatQuantLeNet(bias_quant=False, reduced_act_quant=True, signed_weight=True)

# ... training ...

traced_pt_lenet = PytorchQuantManager.export(pt_lenet, input_shape=(1, 1, 32, 32))
```

Note how the network was parametrized to reflect a few of the differences between PyTorch quantized inference operators and the standard ONNX opset: 
- Pytorch doesn't support bias quantization, standard ONNX does. 
- Both PyTorch and ONNX supports signed symmetric weights, but in practice supports for onnxruntime for the is still limited, so we go with asymmetric weights there. 
- With the FBGEMM x86 backend (which is enabled by default), PyTorch recommends to use 7-bit activations to avoid overflow.

### Export to TVM

The Pytorch export flow generates a TorchScript model, which means that the network can also easily be passed to any external toolchain that supports TorchScript, such as *TVM*:

```python
from tvm import relay

input_name = "input"  
input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(traced_pt_lenet, input_shapes)
```

### Fixed-point quantization for Xilinx DPUs

Thanks to their flexibility, Xilinx FPGAs support a variety of neural network hardware implementations. 
DPUs are a family of fixed-point neural network accelerators officially supported as part of the Vitis-AI toolchain.
Currently Brevitas supports training for DPUv1 and DPUv2 by leveraging fixed-point quantizers and a custom ONNX based export flow that targets PyXIR:

```python
from torch import nn
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int8BiasPerTensorFixedPoint as BiasQuant
from brevitas.export import DPUv1Manager, DPUv2Manager


class DPULeNet(nn.Module):
    def __init__(self):
        super(DPULeNet, self).__init__()
        self.conv1 = qnn.QuantConv2d(
            1, 6, 5, input_quant=ActQuant, weight_quant=WeightQuant,
            output_quant=ActQuant, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = nn.ReLU()
        self.conv2 = qnn.QuantConv2d(
            6, 16, 5, weight_quant=WeightQuant, output_quant=ActQuant,
            bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu2 = nn.ReLU()
        self.fc1   = qnn.QuantLinear(
            256, 120, bias=True, weight_quant=WeightQuant,
            bias_quant=BiasQuant, output_quant=ActQuant, return_quant_tensor=True)
        self.relu3 = nn.ReLU()
        self.fc2   = qnn.QuantLinear(
            120, 84, bias=True, weight_quant=WeightQuant,
            bias_quant=BiasQuant, output_quant=ActQuant, return_quant_tensor=True)
        self.relu4 = nn.ReLU()
        self.fc3   = qnn.QuantLinear(
            84, 10, bias=False, weight_quant=WeightQuant, output_quant=ActQuant)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

dpu_lenet = DPULeNet()
    
# ... training ...

DPUv1Manager.export(dpu_lenet, input_shape=(1, 1, 32, 32), export_path='dpuv1_lenet.onnx')
DPUv2Manager.export(dpu_lenet, input_shape=(1, 1, 32, 32), export_path='dpuv2_lenet.onnx')

```

### Weights-only quantization

Finally, it's also possible to selectively quantize only certain parts of the network at a sub-layer granularity, for example just the weights:

```python
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

import brevitas.nn as qnn


class QuantWeightLeNet(Module):
    def __init__(self):
        super(QuantWeightLeNet, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, weight_bit_width=3)
        self.relu1 = nn.ReLU()
        self.conv2 = qnn.QuantConv2d(6, 16, 5, weight_bit_width=3)
        self.relu2 = nn.ReLU()
        self.fc1   = qnn.QuantLinear(16*5*5, 120, bias=True, weight_bit_width=3)
        self.relu3 = nn.ReLU()
        self.fc2   = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=3)
        self.relu4 = nn.ReLU()
        self.fc3   = qnn.QuantLinear(84, 10, bias=False, weight_bit_width=3)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.reshape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
```

In this case the model cannot be exported anywhere, so it cannot be executed outside of Brevitas. This makes it not particularly interesting from an acceleration point of view, as Brevitas is not optimized for inference.
However, it can still be informative from an accuracy point of view to evaluate different quantization methods and strategies.

## Documentation

Documentation is currently a work-in-progress.  
A series of tutorials is being added to the *notebooks* folder. They are designed to walk users through some of the fundamentals of Brevitas, and as such they are meant to be followed in order.  
A general description of how Brevitas works can be found under the *ARCHITECTURE.md* file.

## Settings

Brevitas exposes a few settings that can be toggled through env variables.

- **BREVITAS_JIT=1** (*Default: = 0*): Enables compilation of the available built-in quantizers through TorchScript just-in-time compiler, 
  together with a small native .cpp extension for the straight-through estimator functions. This can provide a speed-up and/or memory savings at training time. 
  Please note that under certain circumstances this has been shown to produce diverging results compared to BREVITAS_JIT=0. Use at your own risk. 

- **BREVITAS_VERBOSE=1** (*Default: = 0*): Enables verbose compilation of the straight-through estimator functions native extension.

- **BREVITAS_IGNORE_MISSING_KEYS=1** (*Default: =0*): Ignore errors related to missing *state_dict* values when loading a pre-trained model on top of a Brevitas model.
    This is typically enabled when re-training from a floating-point checkpoint.

## F.A.Q.

**Q: Pytorch supports quantization-aware training. Why should I use Brevitas?**

**A:** Quantization in Pytorch is designed to target two specific CPU backends (FBGEMM and qnnpack). 
Export to standard ONNX for quantized operators is not supported (only to a custom ONNX based format supported by the Caffe2).

Brevitas is designed as a platform to implement novel quantization algorithms to target a variety of hardware backends adhering to a loose set of assumptions (i.e. uniform affine quantization).

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
As a general note though, currently FINN is the only toolchain that supports acceleration of low bit-width datatypes.

**Q: My (C/G/T)PU supports float16 / bfloat16 / bfloat19 training. Can I use it to train with Brevitas?**

**A:** Datatypes outside of float32 at training time have not been tested. That includes training on TPU / Pytorch-XLA.
Do the math in terms of which reduced-precision integers can reasonably fit in a reduced-precision 
floating-point format at training time, and use at your own risk.


## Author

Alessandro Pappalardo (@volcacius) @ Xilinx Research Labs.
For private communications, you can reach me at *alessand at name_of_my_employer dot com*.

## Cite as
```
@software{brevitas,
  author       = {Alessandro Pappalardo},
  title        = {Xilinx/brevitas},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3333552},
  url          = {https://doi.org/10.5281/zenodo.3333552}
}
```

