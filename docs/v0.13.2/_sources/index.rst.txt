========
Brevitas
========

.. toctree::
   :maxdepth: 1

   Setup <setup>

   Getting Started <getting_started>

   Tutorials <tutorials/index>

   Papers <papers/index>

   User Guides <user_guide/index>

.. toctree::
   :hidden:

   Settings <settings>
   
   FAQ <faq>

   API reference <api_reference/index>

   About <about>

Brevitas implements a set of building blocks at different levels of abstraction to model a reduced precision hardware data-path at training time.
It provides a platform both for researchers interested in implementing new quantization-aware training techinques, as well as for practitioners interested in applying current techniques to their models.

Brevitas supports a super-set of quantization schemes implemented across various frameworks and compilers under a single unified API.
For certain combinations of layers and types of of quantization inference acceleration is supported by exporting to *FINN*, *onnxruntime* or *Pytorch*'s own quantized operators.

Brevitas has been successfully adopted both in various research projects as well as in large-scale commercial deployments targeting CPUs, GPUs, and custom accelerators running on AMD FPGAs. The general quantization style implemented is affine quantization, with a focus on uniform quantization. Non-uniform quantization is currently not supported out-of-the-box.
