Optional Training Requirements
''''''''''''''''''''''''''''''

- PyTorch compatible C++ compiler

Brevitas implements a small set of custom ``torch.autograd.Function`` both in Python and in C++.

By default, the Python implementation is adopted.The C++ implementation is there to support end-to-end compilation of a training-time quantizer when the jit is enabled (with the env variable BREVITAS_JIT=1).
This is because jit compilation of a custom Python ``torch.autograd.Function`` is currently not supported by PyTorch (as of version 1.8.1).

While end-to-end compilation of a quantizer can provide (very small) benefits in terms of training performances, it's almost never necessary, 
except for some distributed training scenarios with ``BREVITAS_JIT=1``.

.. note::

    The custom C++ autograd functions are implemented in a single `.cpp` file distributed together with Brevitas that is internally compiled and loaded at runtime through the ``torch.utils.cpp_extension.load()`` mechanism.
    This simplifies the packaging of Brevitas, since it avoids to mantain precompiled binaries for every possible supported platform, but it puts on the user the burden of making sure an appropriate compiler is present.


Optional Inference Requirements
'''''''''''''''''''''''''''''''
By design, inference in Brevitas is slow (slower than a corresponding floating-point model would be), as quantized operators are implemented on top of floating-point primitives, with the so-called "fake-quantization" approach.
Efficient quantized inference requires to export a trained quantized model to an inference toolchain that supports that part.

- `FINN`_: for generation of custom FPGA dataflow accelerators.
- `ONNXRuntime`_: for integration with ONNX-based toolchains.
- `Vitis-AI`_: for integration with XIR and PyXIR based toolchains.


.. _Pytorch: https://pytorch.org
.. _FINN: https://xilinx.github.io/finn/
.. _Vitis-AI: https://github.com/Xilinx/Vitis-AI
.. _ONNXRuntime: https://github.com/Microsoft/ONNXRuntime