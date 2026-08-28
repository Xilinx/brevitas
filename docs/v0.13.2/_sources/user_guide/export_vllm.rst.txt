====================
vLLM Export
====================


`vLLM <https://github.com/vllm-project/vllm>`_ is a popular high-throughput and memory-efficient
inference and serving engine for Large Language Models (LLMs).

In Brevitas, we provide an export flow that produces vLLM-compatible quantized models
directly from the LLM entrypoint. The exported model can then be loaded and served through
vLLM with quantization applied transparently.

.. note::

   vLLM is not a dependency of Brevitas and must be installed separately.
   The vLLM import only occurs when ``--export-target vllm`` is specified.


QDQ-based Export
================

The current vLLM export flow is based on a **Quantize-DeQuantize (QDQ)** approach:

* Weights are stored in their dequantized (full precision) form via ``model.save_pretrained()``.
* All quantization metadata (scales, zero points, bit widths, handler class types, rotation
  configurations) is serialized into a ``brevitas_config.json`` file alongside the model.
* At load time, vLLM discovers the ``brevitas_config.json``, reconstructs the quantization
  handlers for each layer, and re-quantizes the weights.

This means the model directory produced by the export is a standard HuggingFace-compatible
checkpoint, with the addition of ``brevitas_config.json`` that vLLM uses to apply quantization.

A **Quantized Operations (QOp)** approach, where operations such as matrix multiplication run
directly on quantized data using accelerated kernels, is currently a work in progress and is
not yet available.


Supported Configurations
========================

Weight Quantization
-------------------

The following weight quantization formats are supported:

* **Integer**: int8 (symmetric), uint8 (asymmetric)
* **FP8**: e4m3 (both ``float`` and ``float_ocp`` variants)
* **MX formats**: MXInt8, MXFloat8e4m3

Weight quantization granularity options:

* Per tensor
* Per channel
* Per group (groupwise), with configurable group size


Activation Quantization
-----------------------

The following activation quantization formats are supported:

* **Integer**: int8 (symmetric), uint8 (asymmetric)
* **FP8**: e4m3 (both ``float`` and ``float_ocp`` variants)
* **MX formats**: MXInt8, MXFloat8e4m3

Activation quantization granularity options:

* Per tensor (static)
* Per row (dynamic) -- FP8 only
* Per group (dynamic)




LLM Entrypoint
==============

Brevitas' LLM entrypoint allows the user to load, quantize, test, and export many of the LLM available on
HuggingFace, by simply passing a series of command line arguments that can control, among other things:

* Weights and activations bit width
* Weights and activation quantization format (int vs float, asym vs sym, etc.)
* PTQ algorithms to apply and their options
* and much more...

Below are some example configurations for the vLLM export target.


**Weight-only INT8 symmetric per-group quantization:**

.. code-block:: bash

   brevitas_ptq_llm --model org/model --weight-bit-width 8 --weight-quant-format int \
     --weight-quant-type sym --weight-quant-granularity per_group --weight-group-size 128 \
     --export-target vllm --export-prefix ./exported_model


**Weight-only INT4 symmetric per-group quantization:**

.. code-block:: bash

   brevitas_ptq_llm --model org/model --weight-bit-width 4 --weight-quant-type sym \
     --weight-quant-granularity per_group --weight-group-size 128 \
     --export-target vllm --export-prefix ./exported_model


**FP8 weight and dynamic per-row activation quantization:**

.. code-block:: bash

   brevitas_ptq_llm --model org/model --weight-bit-width 8 \
     --weight-quant-format float_ocp_e4m3 --weight-quant-granularity per_channel \
     --input-bit-width 8 --input-quant-format float_ocp_e4m3 \
     --input-quant-granularity per_row --input-scale-type dynamic \
     --act-calibration --export-target vllm --export-prefix ./exported_model


These commands will produce quantized models without any extra pre-processing, but several PTQ
algorithms are compatible with this export flow (see below).

The ``--export-prefix`` argument specifies the output directory where the model, tokenizer,
and ``brevitas_config.json`` will be saved.


Compatible PTQ Algorithms
=========================

The following PTQ algorithms are compatible with the vLLM export flow:

* GPTQ / GPFQ
* MagR
* SmoothQuant (fused version only, not layerwise)
* Weight Equalization
* QuaRot / SpinQuant (all rotation modes: ``fx``, ``fused_no_fx``, ``layerwise``)
* Bias Correction
* Qronos

The following are **not** currently supported:

* AWQ
* Learned Round (support is a work in progress)
* SVDQuant


Export Output
=============

When the export completes, the output directory (specified by ``--export-prefix``) will contain:

* **Model weights**: saved via ``model.save_pretrained()`` in standard HuggingFace format.
* **Tokenizer files**: saved via ``tokenizer.save_pretrained()``.
* **brevitas_config.json**: a JSON file containing per-layer quantization metadata, including:

  * Quantization handler class type per proxy (weight, input, output, bias)
  * Scale and zero point values
  * Bit width, float-to-int implementation type, and scaling restriction settings
  * Rotation configurations (Hadamard matrix shape and ``k`` value) for layers wrapped
    in ``RotatedModule``

At load time in vLLM, the registered ``quant_brevitas`` quantization method discovers
``brevitas_config.json`` and assigns a ``QuantLinear`` method to each linear layer. This method
reconstructs the quantization handlers from the saved configuration and applies quantization
during inference.


Using the Exported Model in vLLM
=================================

Once the model has been exported, it can be loaded in vLLM like any other quantized model by
pointing to the export directory:

.. code-block:: python

   from vllm import LLM

   llm = LLM(model="./exported_model", quantization="quant_brevitas")

Or via the vLLM CLI:

.. code-block:: bash

   vllm serve ./exported_model --quantization quant_brevitas


FAQ
===


* *What is the difference between QDQ and QOp?*

QDQ (Quantize-DeQuantize) exports dequantized weights alongside quantization metadata.
At load time, vLLM reconstructs the quantization and re-quantizes the weights.
QOp (Quantized Operations) operates directly on quantized tensors using accelerated kernels
(e.g., MXFP4 via ``aiter``), avoiding the dequantize-requantize overhead.
Currently only the QDQ approach is available; QOp is a work in progress.


* *Can I use custom quantizers with the vLLM export?*

Yes, the ``--custom-quantizer`` flag is compatible with the vLLM export path, as long as
the underlying quantization format maps to one of the supported inference handlers.


* *Why do I get an import error for vLLM?*

vLLM is not bundled with Brevitas and must be installed separately in your environment.
The vLLM-specific code is only imported when ``--export-target vllm`` is specified, so
vLLM is not required for other Brevitas workflows.
