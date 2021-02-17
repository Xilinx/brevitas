# Brevitas

[![Gitter](https://badges.gitter.im/xilinx-brevitas/community.svg)](https://gitter.im/xilinx-brevitas/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
![Pytest](https://github.com/Xilinx/brevitas/workflows/Pytest/badge.svg?branch=master)
![Examples Pytest](https://github.com/Xilinx/brevitas/workflows/Examples%20Pytest/badge.svg?branch=master)
[![DOI](https://zenodo.org/badge/140494324.svg)](https://zenodo.org/badge/latestdoi/140494324)

Brevitas is a Pytorch library for quantization-aware training (QAT).

*Brevitas is currently under active development. Documentation, tests, examples, and pretrained models will be progressively released.*

**Please note that Brevitas is a research project and not an official Xilinx product.**

## History

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

## Architecture
Brevitas is organized around a few different concepts listed below.

### Functions
Functions include both Autograd functions and TorchScript functions and can be found under `brevitas.function`. 

Autograd functions implement straight-through estimators. 
All Autograd functions are implemented both in Python, found under `brevitas.function.autograd_ste_ops`,
and C++, found under _brevitas/csrc_. 
This is because to date Python Autograd functions cannot be compiled
by Pytorch's JIT compiler, but C++ Autograd functions can. Compiling the C++ extension is performed at runtime
in order to simplify packaging and distribution (only a single .cpp source file is distributed). 
This requires the user to have an appropriate C++ compiler, so a Python implemention is provided as fallback.
Making sure that autograd functions can be compiled is relevant when `BREVITAS_JIT=1` is set in a 
`DistributedDataParallel` training setting. 
Because of this, as long as `BREVITAS_JIT=0` (which it is by default), the C++ backend is not loaded.
Wrapping and switching between the two implementations happens in `brevitas.function.ops_ste`.

TorchScript functions implements various bits and pieces used commonly across Brevitas that are 
required to be JIT-compilable. They can be found under `brevitas.function.ops` and `brevitas.function.shape`. 

### Core ScriptModules
The algorithmic core of Brevitas can be found under `brevitas.core`.
The core implements all the various building blocks required to assemble an affine quantizer. 
All building blocks are implemented as a ScriptModule in the _old-style_ TorchScript scriping API 
(i.e. inheriting from `torch.jit.ScriptModule`). Many of the functions described in the section above
are called from within one of the core ScriptModules.
Any module within `brevitas.core` is required to be a ScriptModule compatible with Pytorch 1.3.0 JIT compiler,
with most building blocks being compatible with Pytorch 1.1.0 JIT compiler. 
Implementing everything in TorchScript enables compute and memory optimizations at training time, 
which for more complicated quantization pipelines can be quite significant, thus reducing the intrinsic 
training-time cost of quantization-aware training.

Adhering to the restriction imposed by TorchScript pose a challange in terms of how to achieve flexibility
while minimizing redundancy. Because the flavour of TorchScript adopted by Brevitas does not allow 
for inheritance, Brevitas' core is highly biased towards leveraging composition. 
In particular, the implementation favours inversion of control through dependency injection (DI), 
whether manually or through the DI library _dependencies_ (as explained in the Injectors section below.)

### Injectors and Quantizers
Auto-wiring dependency injection (DI) with _dependencies_ is the machinery at the heart of Brevitas. 
If you have ever used fixtures in pytest, you already know (high-level) how auto-wiring DI works. 
In the case of _dependencies_, the idea is that the objects to be instantiated and wired together are
declared as attributes of an Injector class. The driving mechanism behind the auto-wiring process is
to match the name of an attribute of an Injector with them name of arguments required to build other 
attributes of the same Injectors. \
When applied to Brevitas then, the idea is to throw together a bunch of components 
appropriately chosen from `brevitas.core` as attributes of an Injector, and have the DI library automatically
assemble them together. A quantizer then is an Injector with a `tensor_quant` attribute. 
A `tensor_quant` object is expected to be a torch _Module_ (either a `torch.nn.Module` or a `torch.jit.ScriptModule`) 
that takes in as input a torch.Tensor to quantize, and return as output a tuple containing four 
`torch.Tensor`, representing respectively the output _quantized tensor_ in _dequantized format_, 
its _scale factor_, its _zero-point_, and it's _bit-width_.

Injectors are a powerful way to express quantizers: being standard Python classes, they lend themselves
to both inheritance and composition (through multiple inheritance in a mixin style). 
That means for example that a new quantizer can be defined simply by inheriting from an existing one 
and overriding some of its attributes, or by inheriting from multiple smaller Injector that 
declare only some of the components required to assemble a quantizer. Pre-built quantizers can be found under
`brevitas.quant`. Specifically, `brevitas.quant.shifted_scaled_int` holds quantizers with _zero-point != 0._
and _floating-point_ scale factors, `brevitas.quant.scaled_int` holds quantizers with _zero-point == 0._ and
_floating-point_ scale factors, and `brevitas.quant.fixed_point` holds fixed-point, i.e. quantizers
with _zero-point == 0_ and _power-of-two_ scale factors. Pre-built quantizers are assembled together
from smaller Injectors that can be found in `brevitas.quant.base`.

Brevitas depends on a specific older version of _dependencies_ (_v2.0.1_) that plays well with ScriptModules,
and it extends it with an `ExtendedInjector` class that provides support for some additional features.
Specifically, `ExtendedInjector` allows to dynamically declare an object to auto-wire by returning a class
from a `@value` function. This style of syntax allows to create intermediate abstractions between the
`brevitas.core` components and the `brevitas.quant` quantizers, as explained in the section below.

### Enums, Shapes and Solvers
Trying to navigate all the various components in `brevitas.core` that can be assembled together in a 
quantizer can be intimidating. The lack a of clear object hierarchy between various ScriptModules means 
that it's not obvious which components adhere to a certain interface. Additionally, while some components 
might fit together from a purely software perspective, they might not work together
from a machine learning perspective. In order to provide a simplified interface that abstracts away some
of these details, Brevitas provides various `Enum` on top of the components found in `brevitas.core`.\
Some examples of enums are: `QuantType` specifies the type of quantization, e.g. `QuantType.BINARY` 
for binary quantization. `ScalingImplType`, for specifying the type (algorithmic-wise) of scaling, 
e.g. `ScalingImplType.PARAMETER_FROM_STATS` to specify that the scale factor should be a learned 
`torch.nn.Parameter` initialized from statistics of the tensor to quantize. 
Enums can currently be found under `brevitas.inject.enum`.\
Depending on the kind of tensor to quantize, say _weights vs activations_, the same enum value is gonna
translate to different `brevitas.core` components. So for example `QuantType.BINARY` translates to
`brevitas.core.quant.BinaryQuant` for weights and to `brevitas.core.quant.ClampedBinaryQuant` for activations.
This way enums allows to declare quantizers while abstract away from the specifics of a target tensor. 
In general there can be a _1-to-1_, _many-to-1_ or _many-to-many_ relationship between enums and 
`brevitas.core` components and their hyperparameters.

The translation between enums and `brevitas.core` is performed by a *solver*, which can be found under
`brevitas.quant.solver`.\
Solvers are really just an `ExtendedInjector` that take advantage of the extended syntax of Injectors implemented in Brevitas
to translate e.g. `quant_type = QuantType.BINARY` to `tensor_quant = BinaryQuant` within the scope of 
a quantizer at _dependency-injection time_. That means that solvers are as composable as quantizers are,
so for example to solve enums against weight quantization, it's enough to create a quantizer that inherits from 
`brevitas.quant.solver.weight.WeightQuantSolver`, which is itself just the collection of various solvers for individual tasks.
Looking at the quantizers found under brevitas.quant, it can be seen that for the most part they actually specify
enums rather than directly `brevitas.core` components. Then enums are then solved to different core components
depending on which solver is applied. This is meant to provide a blueprint for users to understand which enums
are supported and how they go together.

A similar mechanism applies when for example the directive `scaling_per_output_channel=True` is specified. 
In the case of `ScalingImplType.PARAMETER_FROM_STATS`, that means that a `torch.nn.Parameter` with 
size the number of channels of the tensor to quantize has to be allocated. Because TorchScript does not allow for much dynamic behaviour, 
the shape of the parameter has to be known at dependency-injection time. Rather than forcing the user
to specify the appropriate shape in a quantizer, Brevitas is capable of inferring it from the `nn.Module` 
whose weights the quantizer is applied to. This again happens by means of a solver included as part of 
`WeightQuantSolver`.

Thanks to how dependencies works, solvers are invoked only whenever their output is actually required to build
an attribute of an `ExtendedInjector`, which in the case of Brevitas is `tensor_quant`. Additionally,
by specifying a solver as last in the list of classes from which a quantizer inherits from,
it's always possible to override its behaviour and directly declare its output. So for example it's
possible to directly declare `tensor_quant = BinaryQuant` instead of `quant_type = QuantType.BINARY` even
when `WeightQuantSolver` is applied to a quantizer. This allows more advanced users to mix-and-match
enums with custom components. Finally, it's always possibly to just not apply a solver to a quantizer
and simply declare everything manually.

### QuantTensor
A `QuantTensor` is a custom data structure for representing a uniform, affine quantized tensor. 
It can be found under `brevitas.quant_tensor`. 
It can be _valid_ or _non-valid_. A non-valid `QuantTensor` is simply a wrapper around a torch.Tensor that 
had been previously quantized and is now in dequantized format. The `QuantTensor` is marked as non-valid because it doesn't carry enough information
to derive its quantized representation back. A valid QuantTensor carries _scale_, _zero-point_, _bit-width_, _sign_, _narrow-range_,
and whether it was generated in training or inference mode.

The arithmetic of QuantTensors implments a generalized version of fixed-point arithmetic, with the main assumption
being that only two QuantTensor with the same scale factor can be summed together. This constrain is enforced
when the QuantTensors involved in a sum have been generated in inference mode, but it's not enforced in training mode.
This is because when dealing with e.g. `ScalingImplType.PARAMETER_FROM_STATS`, the activation tensors in a residual topology
can have different scale factors along the skip connection at training time, but not at inference time.

### Proxies
A proxy is a `nn.Module` that wraps a quantizer. Proxies can be found under `brevitas.proxy`. 
Proxies are specialized w.r.t. the kind of tensor they quantize, such as _weights_, _biases_, or _activations_.

The main responsability of a proxy is to make sure a QuantTensor is returned as output of quantization, which wouldn't be
possible in TorchScript. Additionally, it has to make sure a quantizer is re-initialized any time it is necessary.\
For example, when performing `ScalingImplType.PARAMETER_FROM_STATS` scaling on a weight tensor, 
statistics of the weight tensor are computed at dependency-injection time and used to initialize a 
learned parameter. However, it's not possible to know a-priori whether a pretrained floating-point state-dict 
will be later on loaded on top of a quantized model definition. In that case, any initialization logic that 
depends on the state-dict of the model that is being quantized has to be recomputed.
Injectors invoked by proxies on state_dict changes allow to do so automatically, 
providing a mechanism to reconcile the inherent rigidity of TorchScript with the typical define-by-run execution of Pytorch models.  

Proxies also allow to support more complex quantization scenarios, such as when the same quantizer has to be shared between
different layers. A typical situation where that happens is when the output of multiple branches of a residual topology are summed together
at high-precision, without requantizing first. In that case, the weight tensors of the layers that feed into the accumulator
need to have the same scale factor. That can be accomplished by declaring a single `WeightQuantProxy` that is shared among multiple layers.
What happens is that - for example for `ScalingImplType.STATS`, `ScalingImplType.AFFINE_STATS` or `ScalingImplType.PARAMETER_FROM_STATS` -
the scale factor is computed as a statistics of the concatenation of the weight tensors to be quantized. 
Because it can't be known a-priori to between how many layers the same WeightQuantProxy is shared, again every
time a WeightQuantProxy starts tracking a new weight tensor, the underlying quantizer has to be re-initialized.

### Quant Layers and Mixins
A `QuantLayer` is the quantized variant of a `torch.nn` layer, and can be found under `brevitas.nn`.
Typically a QuantLayer inherits from both its floating-point variant (e.g. `QuantConv2d` inherits from `Conv2d`),
plus a serie of mixins, each responsibile for instantiating a proxy within the QuantLayer. 
A mixin is more specialized than a proxy, so for example both `QuantInputMixin` and `QuantOutputMixin` 
instantiate an activation quantization proxy. So-called _QuantWBIOL_ layers (such as `QuantConv2d`)
inherit from `QuantWeightMixin`, `QuantBiasMixin`, `QuantInputMixin` and `QuantOutputMixin`.
That means that they can quantize respectively _weight_, _bias_, _input_ and _output_. 
Quantizers can be passed to a QuantWBIOL layer by setting respectively `quant_weight=`, `quant_bias=`, `quant_input=`
and `quant_output=`. If an `ExtendedInjector` is passed it, a proxy will be allocated by the mixin to deal with its
initialization. Otherwise, if a proxy is passed in, it will be set as-is. Setting e.g. `quant_weight=None`
will disable quantization for weights. A layer where quantization is disabled is supposed to act exactly like
its floating-point counterpart, so `QuantConv2d(..., weight_quant=None, bias_quant=None, input_quant=None, output_quant=None)`
behaves like a `Conv2d` layer.

In many real-life scenarios, a user might want to first quantize only certain layers or certain parts 
of certain layers to perform some exploratory analysis in terms of accuracy. 
Correctness w.r.t. the specifics of a target hardware, if any, might not be a concern. To minimize friction with adoption then, 
Brevitas is designed to remain functional as much as possible under partially specified information.
With the exception of `QuantAvgPool2d`, a `QuantLayer` is not expected to receive a `QuantTensor` as input 
(altough doing so enables more scenarios), nor it returns one by default (i.e. `return_quant_tensor=False` by default).
Specifically, The output of a QuantLayer is always in de-quantized format 
(whether wrapped in a valid or non-valid QuantTensor or not).
That means that QuantLayers can be easily mixed with standard `torch.nn` layers. 


### Export 
### Graph tracing and transformations _(experimental)_
### Losses

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


