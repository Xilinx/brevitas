======
F.A.Q.
======

**Q: Pytorch supports quantization-aware training. Why should I use
Brevitas?**

**A:** Quantization in Pytorch is designed to target two specific CPU
backends (FBGEMM and qnnpack). Export to standard ONNX for quantized
operators is not supported (only to a custom ONNX based format supported
by the Caffe2).

Brevitas is designed as a platform to implement novel quantization
algorithms to target a variety of hardware backends adhering to a loose
set of assumptions (i.e. uniform affine quantization).

**Q: How can I train X/Y and run it on hardware W/Z? I can't find any
documentation.**

**A:** Brevitas is still sparsely documented. Until the situation
improves, feel free to open an issue or ask on our gitter channel.

**Q: Training with Brevitas is slow and/or I can't fit the same batch
size as with floating-point training. Why? What can I do?**

**A:** Quantization-aware training involves a lot of element-wise
operations, which carry low arithmetic intensity and contribute to a
more involved computational graph during backpropragation. As such, it
typically ends up being slower and more resource-intensive than standard
floating-point training.

Brevitas in particular is biased towards greater flexibility, at the
cost of some training-time effieciency. The general principle is that
it's trading off more complexity at training time for more efficiency at
inference time.

To mitigate somewhat the slow-down, try enabling *BREVITAS_JIT* as
reported in the *Settings* section.

**Q: Inference with Brevitas is slow. I thought the point of QAT was to
make my model faster at inference time. What I am doing wrong?**

**A:** Brevitas is concerned with modelling a reduced precision
data-path, it does not provide inference-time acceleration on its own.
To achieve acceleration, you should export your Brevitas model to a
downstream toolchain / backend.

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

**Q: My (C/G/T)PU supports float16 / bfloat16 / bfloat19 training. Can I
use it to train with Brevitas?**

**A:** Datatypes outside of float32 at training time have not been tested. That includes training on TPU / Pytorch-XLA.
Do the math in terms of which reduced-precision integers can reasonably fit in a reduced-precision
floating-point format at training time, and use at your own risk.