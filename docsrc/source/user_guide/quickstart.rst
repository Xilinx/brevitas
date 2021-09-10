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