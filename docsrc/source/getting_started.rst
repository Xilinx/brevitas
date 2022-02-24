===============
Getting started
===============

Brevitas serves various types of users and end goals. To showcase some
of Brevitas features, we consider then different scenarios for the
quantization of a classic neural network, LeNet-5.

Weights-only quantization
-------------------------

Let's say we are interested in assessing how well the model does at *3
bit weights* for CIFAR10 classification. For the purpose of this
tutorial we will skip any detail around how to perform training, as
training a neural network with Brevitas is no different than training
any other neural network in PyTorch.

``brevitas.nn`` provides quantized layers that can be used **in place
of** and/or **mixed with** traditional ``torch.nn`` layers. In this case
then we import ``brevitas.nn.QuantConv2d`` and
``brevitas.nn.QuantLinear`` in place of their PyTorch variants, and we
specify ``weight_bit_width=3``. For relu and max-pool, we leverage the
usual ``torch.nn.ReLU`` and ``torch.nn.functional.max_pool2d``.

The result is the following:

.. code:: python

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

   quant_weight_lenet = QuantWeightLeNet()

   # ... training ...

At the end of training the model is going to have a certain train and
test accuracy. For users interested in simply evaluating how well their
models do with quantization in the loop, without actually deploying
them, that might be the end of it.

For those users that instead are interested in deploying their quantized
models, the idea obviously would be to actually gain some kind of
advantage from quantization. In the case of weight quantization, the
advantage would be to save space in terms of model size. However, if we
saved the model state with
``torch.save(quant_weight_lenet.state_dict(), 'qw_lenet.pt')`` we would
notice that it consumes the same amount of memory as its floating-point
variant. That is because Brevitas is not concerned with deploying
quantized models efficiently on its own.
In order to deploy the model efficiently, we have to export it to an
inference framework/toolchain first.

Being a research training library that informs the development of
inference toolchains, Brevitas supports more quantization schems than
what can be currently accelerated efficiently by supported inference
frameworks. A neural network with 3 bits weights and floating-point
activations is one of those scenarios that in practice is currently hard
to take advantage of. In order to make it practical, we want to quantize
activations and biases too.

Low-precision integer-only LeNet
--------------------------------

We decide to quantize activations to 4 bits and biases to 8 bits. In
order to do so, we replace ``torch.nn.ReLU`` with
``brevitas.nn.QuantReLU``, specifying ``bit_width=4``. For bias
quantization, we import the 8-bit bias quantizer ``Int8Bias`` from
``brevitas.quant`` and set it appropriately. Additionally, in order to
quantize the very first input, we introduce a
``brevitas.nn.QuantIdentity`` at the beginning of the network. The end
result is the following:

.. code:: python

   from torch.nn import Module
   import torch.nn.functional as F

   import brevitas.nn as qnn
   from brevitas.quant import Int8Bias as BiasQuant


   class LowPrecisionLeNet(Module):
       def __init__(self):
           super(LowPrecisionLeNet, self).__init__()
           self.quant_inp = qnn.QuantIdentity(
               bit_width=4, return_quant_tensor=True)
           self.conv1 = qnn.QuantConv2d(
               3, 6, 5, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
           self.relu1 = qnn.QuantReLU(
               bit_width=4, return_quant_tensor=True)
           self.conv2 = qnn.QuantConv2d(
               6, 16, 5, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
           self.relu2 = qnn.QuantReLU(
               bit_width=4, return_quant_tensor=True)
           self.fc1   = qnn.QuantLinear(
               16*5*5, 120, bias=True, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
           self.relu3 = qnn.QuantReLU(
               bit_width=4, return_quant_tensor=True)
           self.fc2   = qnn.QuantLinear(
               120, 84, bias=True, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
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

Note a couple of things:

- Compared to the previous scenario, we now set ``return_quant_tensor=True`` in every quantized layer except the last one to propagate a ``QuantTensor`` across them. This informs each receiving layer of how activations have been quantized at the output of its predecessor, which in turns enables more functionalities, such as the kind of bias quantization here implemented.
- ``torch`` operations that are algorithmically invariant to quantization, such as max-pool, can propagate QuantTensor through them without extra changes. This is supported in PyTorch 1.5.0 and later versions.
- By default ``QuantReLU`` is *stateful*, so there is a difference between instantiating one ``QuantReLU`` that is called multiple times, and instantiating multiple `QuantReLU` that are each called once.

Dataflow FPGA acceleration with FINN
------------------------------------

The network defined above can be mapped to a low-precision *integer-only* dataflow accelerator implemented on a Xilinx FPGA by exporting it to FINN through a custom ONNX-based representation. We can invoke the FINN export manager to do so:

.. code:: python

    from brevitas.export import FINNManager

    low_precision_lenet = LowPrecisionLeNet()

    # ... training ...

    FINNManager.export(low_precision_lenet, input_shape=(1, 3, 32, 32), export_path='finn_lenet.onnx')


A mixed float-integer LeNet
---------------------------

Brevitas also supports targeting other inference frameworks that support a mixture of floating-point and quantized layers, such as *onnxruntime* and *PyTorch* itself.
In this case then, ``return_quant_tensor`` clarifies to the export manager whether the output of a layer should be dequantized to floating-point or not.
Additionally, since for those target platforms low precision acceleration is not yet supported, we target 7-bit and 8-bit quantization:

.. code:: python

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
                3, 6, 5, input_quant=act_quant, weight_quant=weight_quant,
                output_quant=act_quant, bias_quant=bias_quant, return_quant_tensor=True)
            self.relu1 = nn.ReLU()
            self.conv2 = qnn.QuantConv2d(
                6, 16, 5, weight_quant=weight_quant, output_quant=act_quant,
                bias_quant=bias_quant, return_quant_tensor=True)
            self.relu2 = nn.ReLU()
            self.fc1   = qnn.QuantLinear(
                16*5*5, 120, bias=True, weight_quant=weight_quant,
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


Compared to the previous case, there are a few differences:
- While in the previous example the default *scaled integer* quantized was being adopted for weights and activations, and only a bit-width was specified, here the network is compeltely parametrized by standalone quantizers taken from `brevitas.quant`. This is to match the quantization schemes supported by the inference frameworks we are targeting.
- We are defining a 7-bit activation quantizer by inheriting from an existing one and setting ``bit_width=7``. This is an alternative but equivalent syntax to setting the attribute as a keyword argument.
- In this scenario activations are quantized before *relu* by setting output quantizers on ``QuantConv2d`` and ``QuantLinear``. Again this matches how the frameworks we are targeting work. Because of this, we revert to using standard `torch.nn.ReLU`.


Export to standard ONNX
-----------------------

After training, the above network can then be exported to an ONNX representation that complies with the [standard opset](https://github.com/onnx/onnx/blob/master/docs/Operators.md):

.. code:: python

    from brevitas.export import StdQOpONNXManager

    onnx_lenet = MixedFloatQuantLeNet()

    # ... training ...

    StdQOpONNXManager.export(onnx_lenet, input_shape=(1, 3, 32, 32), export_path='onnx_lenet.onnx')


Acceleration with onnxruntime
-----------------------------

The generated output model can then be accelerated through any ONNX-compliant inference framework, such as *onnxruntime*:

.. code:: python
    import onnxruntime as rt
    import numpy as np

    sess = rt.InferenceSession('onnx_lenet.onnx')
    input_name = sess.get_inputs()[0].name
    pred_onx = sess.run(None, {input_name: np.random.randn(1, 3, 32, 32)})[0]


Export to PyTorch quantized inference ops
-----------------------------------------


With the same network definition it's also possible to target [PyTorch's own quantized inference operators](https://pytorch.org/docs/stable/torch.nn.quantized.html):

.. code:: python
    from brevitas.export import PytorchQuantManager

    pt_lenet = MixedFloatQuantLeNet(bias_quant=False, reduced_act_quant=True, weight_signed=True)

    # ... training ...

    traced_pt_lenet = PytorchQuantManager.export(pt_lenet, input_shape=(1, 3, 32, 32))


Note how the network was parametrized to reflect a few of the differences between PyTorch quantized inference operators and the standard ONNX opset:
- Pytorch doesn't support explicit bias quantization, standard ONNX does.
- We pick an 8-bit signed symmetric weights quantizer for PyTorch (the one used by default for weight quantization in Brevitas), while for ONNX we go for an unsigned asymmetric one, since support for it in onnxruntime is more mature.
- With the FBGEMM x86 backend (which is enabled by default), PyTorch recommends to use 7-bit activations to avoid overflow.


Export to TVM
-------------

The PyTorch export flow generates a TorchScript model, which means that the network can also easily be passed to any external toolchain that supports TorchScript, such as *TVM*:

.. code:: python
    from tvm import relay

    input_name = "input"
    input_shapes = [(input_name, (1, 3, 224, 224))]
    mod, params = relay.frontend.from_pytorch(traced_pt_lenet, input_shapes)


Fixed-point quantization for Xilinx DPUs
----------------------------------------

Thanks to their flexibility, Xilinx FPGAs support a variety of neural network hardware implementations.
DPUs are a family of fixed-point neural network accelerators officially supported as part of the Vitis-AI toolchain.
Currently Brevitas supports training for DPUs by leveraging 8-bit fixed-point quantizers and a custom ONNX based export flow that targets PyXIR:

.. code:: python
    from torch import nn
    import torch.nn.functional as F

    import brevitas.nn as qnn
    from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
    from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
    from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling as BiasQuant
    from brevitas.export import PyXIRManager


    class DPULeNet(nn.Module):
        def __init__(self):
            super(DPULeNet, self).__init__()
            self.conv1 = qnn.QuantConv2d(
                3, 6, 5, input_quant=ActQuant, weight_quant=WeightQuant,
                output_quant=ActQuant, bias_quant=BiasQuant, return_quant_tensor=True)
            self.relu1 = nn.ReLU()
            self.conv2 = qnn.QuantConv2d(
                6, 16, 5, weight_quant=WeightQuant, output_quant=ActQuant,
                bias_quant=BiasQuant, return_quant_tensor=True)
            self.relu2 = nn.ReLU()
            self.fc1 = qnn.QuantLinear(
                16*5*5, 120, bias=True, weight_quant=WeightQuant,
                bias_quant=BiasQuant, output_quant=ActQuant, return_quant_tensor=True)
            self.relu3 = nn.ReLU()
            self.fc2 = qnn.QuantLinear(
                120, 84, bias=True, weight_quant=WeightQuant,
                bias_quant=BiasQuant, output_quant=ActQuant, return_quant_tensor=True)
            self.relu4 = nn.ReLU()
            self.fc3 = qnn.QuantLinear(
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

    PyXIRManager.export(dpu_lenet, input_shape=(1, 3, 32, 32), export_path='pyxir_lenet.onnx')