====================
Export Compatibility
====================

In order to accelerate a quantized model, Brevitas requires to export the model to an inference toolchain. 
Brevitas can currently export to:

-  FINN - for dataflow acceleration on Xilinx FPGAs.
-  PyXIR (*experimental*) - for DPU acceleration on Xilinx FPGAs.
-  Standard ONNX (*experimental*) - for acceleration with e.g.
   onnxruntime, or any other ONNX-compliant toolchain.
-  Pytorch's *quantized.functional* operators (*experimental*) - for
   acceleration through Pytorch itself, or any additional downstream
   toolchains supported by Pytorch (e.g. TVM).

Because Brevitas implements a super-set of layers and datatypes
supported by various downstream toolchains and hardware platforms, the
result is that each export flow supports only a certain subset of
features, in ways that are not necessarely obvious. More examples and
documentation will be released to illustrate the various restrictions
imposed by each target platform. As a general note though, currently
FINN is the only toolchain that supports acceleration of low bit-width
datatypes.

Layer / Toolchain
_________________

Quantized linear layers
'''''''''''''''''''''''


