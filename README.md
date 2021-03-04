# Brevitas

[![Gitter](https://badges.gitter.im/xilinx-brevitas/community.svg)](https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
![Pytest](https://github.com/Xilinx/brevitas/workflows/Pytest/badge.svg?branch=master)
![Examples Pytest](https://github.com/Xilinx/brevitas/workflows/Examples%20Pytest/badge.svg?branch=master)
[![DOI](https://zenodo.org/badge/140494324.svg)](https://zenodo.org/badge/latestdoi/140494324)

Brevitas is a Pytorch library for quantization-aware training (QAT).

*Brevitas is currently under active development. Documentation, tests, examples, and pretrained models will be progressively released.*

**Please note that Brevitas is a research project and not an official Xilinx product.**

## History
- *2021/03/01* - Release version 0.3.1, fix bug w/ act initialization from statistics w/ IGNORE_MISSING_KEYS=1.
- *2021/03/01* - Release version 0.3.0, implements enum and shape solvers within extended dependency injectors. This allows declarative quantizers to be self-contained.
- *2021/02/04* - Release version 0.2.1, includes various bugfixes of QuantTensor w/ zero-point.
- *2021/01/30* - First release version 0.2.0 on PyPI.

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

Brevitas provides a platform both for researchers interested in implementing new quantization-aware training techinques, as well as for practitioners interested in applying current techniques to their models.\
Brevitas has been successfully adopted both in various research projects as well as in large-scale commercial deployments targeting custom accelerators.\
The general quantization style implemented is affine quantization, with a focus on uniform quantization. Non-uniform quantization is currently not supported out-of-the-box.

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

## Documentation

Documentation is currently a work-in-progress.  
A series of tutorials for beginners is being added to the *notebooks* folder.  The first two go through the fundamentals of how quantized layers work.  
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

