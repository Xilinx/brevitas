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
Quantizers can be passed to a _QuantWBIOL_ layer by setting respectively `quant_weight=`, `quant_bias=`, `quant_input=`
and `quant_output=`. If an `ExtendedInjector` is passed it, a proxy will be allocated by the mixin to deal with its
initialization. Otherwise, if a proxy is passed in, it will be set as-is. Setting e.g. `quant_weight=None`
will disable quantization for weights. A layer where quantization is disabled is supposed to act exactly like
its floating-point counterpart, so `QuantConv2d(..., weight_quant=None, bias_quant=None, input_quant=None, output_quant=None)`
behaves like a `Conv2d` layer.

Typically `torch.nn` layers expose a flat interface. To support a similar _UX_ in Brevitas, QuantLayers 
support setting attributes in a quantizer by passing keyword arguments with an appropriate prefix. 
For _QuantWBIOL_ layers, keyword arguments with prefix `weight_` are passed to the `weight_quant` quantizer, `bias_` to `bias_quant`,
`input_` to `input_quant`, and `output_` to `output_quant`.
For `quantized activation` layers, like `QuantReLU`, a prefix is not required, and keyword arguments are directly passed to output quantization.
In case an `ExtendedInjector` is not set, e.g. if `weight_quant=None`, but additional keyword arguments are passed in,
an empty `ExtendedInjector` is automatically allocated and keyword arguments are set as its attribute according to the their prefix.
Keyword arguments have priority over pre-existing attribute of a quantizer, so passing a keyword argument is a way
to override the attribute of a quantizer on an individual-layer level.

In many real-life scenarios, a user might want to first quantize only certain layers or certain parts 
of certain layers to perform some exploratory analysis in terms of accuracy. 
Correctness w.r.t. the specifics of a target hardware, if any, might not be a concern. To minimize friction with adoption then, 
Brevitas is designed to remain functional as much as possible under partially specified information.
With the exception of `QuantAvgPool2d`, a `QuantLayer` is not expected to receive a `QuantTensor` as input 
(altough doing so enables more scenarios), nor it returns one by default (i.e. `return_quant_tensor=False` by default).
Specifically, The output of a QuantLayer is always in de-quantized format 
(whether wrapped in a valid or non-valid `QuantTensor` or not).
That means that QuantLayers can be easily mixed with standard `torch.nn` layers. 


### Export 
### Graph tracing and transformations _(experimental)_
### Losses