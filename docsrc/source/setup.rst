=====
Setup
=====

Requirements
------------

Installation Requirements
'''''''''''''''''''''''''

-  Python >= 3.6.
-  `Pytorch`_ >= 1.1.0 (minimal), >= 1.5.0 (suggested).
-  Windows, Linux or macOS.
-  GPU training-time acceleration (*optional* but recommended).

.. note::

    Brevitas is mostly implemented in a mixture of Python and TorchScript, a subset of Python supported by PyTorch just-in-time compiler.
    All quantization primitives are based on standard aTen operators. That means that Brevitas comes with out-of-the-box support for any PyTorch aTen-based training backend, such as the recently introduced AMD ROCm one.


Supported PyTorch versions
''''''''''''''''''''''''''

.. csv-table::
   :header: "PyTorch version", "Brevitas version"

   "1.1.0", "0.4.0"
   "1.2.0", "0.4.0"
   "1.3.1", "0.4.0"
   "1.4.0", "0.4.0"
   "1.5.0", "0.4.0"
   "1.6.0", "0.4.0"
   "1.7.1", "0.4.0"
   "1.8.1", "master"


Optional Training Requirements
''''''''''''''''''''''''''''''

- PyTorch compatible C++ compiler

Brevitas implements a small set of custom torch.autograd.Function both in Python and in C++.

By default, the Python implementation is adopted.The C++ implementation is there to support end-to-end compilation of a training-time quantizer when the jit is enabled (with the env variable BREVITAS_JIT=1).
This is because jit compilation of a custom Python torch.autograd.Function is currently not supported by PyTorch (as of version 1.8.1).

While end-to-end compilation of a quantizer can provide (very small) benefits in terms of training performances, it's almost never necessary.
The only scenario when it's a requirement is when DistributedDataParallel training is performed with the jit enabled.

.. note::

    The custom C++ autograd functions are implemented in a single .cpp file distributed together with Brevitas that is internally compiled and loaded at runtime through the torch.utils.cpp_extension.load() mechanism.
    This simplifies the packaging of Brevitas, since it avoids to mantain precompiled binaries for every possible supported platform, but it puts on the user the burden of making sure an appropriate compiler is present.

Optional Inference Requirements
'''''''''''''''''''''''''''''''
By design, inference in Brevitas is slow (slower than a corresponding floating-point model would be), as quantized operators are implemented on top of floating-point primitives, with the so-called "fake-quantization" approach.
Efficient quantized inference requires to export a trained quantized model to an inference toolchain that supports that part.

- FINN: for generation of custom FPGA dataflow accelerators
- onnx: for integration with ONNX-based toolchains
- Vitis AI: for integration with XIR and PyXIR based toolchains

Installation
------------

Installing from PyPI
''''''''''''''''''''

You can install the latest release from PyPI:

.. code:: bash

   pip install brevitas

Installing from Github
''''''''''''''''''''''

To get the very latest version, you can install directly from GitHub:

.. code:: bash

   pip install git+https://github.com/Xilinx/brevitas.git

.. _Pytorch: https://pytorch.org




