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


First and Last Layer Quantization
---------------------------------

With ``--quantize-first-last-layer``, the weights of the input embedding and the weights and inputs
of the LM head are quantized in vLLM as well.


Attention Quantization
----------------------

Static per-tensor quantization of the attention query, key, and value is supported with
``--quant-sdpa eager`` and ``--attn-quant-config`` set to ``kv`` or ``qkv``. Keys and values are
quantized before being written to the KV cache. The quantization of the softmax output
(``--attn-quant-config qkvs``) is not supported, since vLLM attention kernels never materialize it.



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
* Expanded rotations (``--rotation-layers-to-expand``)

Block rotations (``--rotation-block-size``) are supported for both fused and online rotations.


Export Output
=============

When the export completes, the output directory (specified by ``--export-prefix``) will contain:

* **Model weights**: saved via ``model.save_pretrained()`` in standard HuggingFace format.
* **Tokenizer files**: saved via ``tokenizer.save_pretrained()``.
* **brevitas_config.json**: a JSON file containing per-layer quantization metadata, including:

  * Quantization handler class type per proxy (weight, input, output, bias)
  * Scale and zero point values
  * Bit width, float-to-int implementation type, and scaling restriction settings
  * Rotation configurations (Hadamard matrix shape, ``k`` value, and block size) for layers
    wrapped in ``RotatedModule``

At load time in vLLM, the registered ``quant_brevitas`` quantization method discovers
``brevitas_config.json`` and assigns a quantization method to each linear, embedding, and attention
layer. This method reconstructs the quantization handlers from the saved configuration and applies
quantization during inference.


Using the Exported Model in vLLM
=================================

Brevitas registers the ``quant_brevitas`` quantization method through a vLLM plugin, which is
installed together with Brevitas. Once the model has been exported, it can be loaded in vLLM like
any other quantized model by pointing to the export directory:

.. code-block:: python

   from vllm import LLM

   llm = LLM(model="./exported_model", quantization="quant_brevitas", enforce_eager=True)

Or via the vLLM CLI:

.. code-block:: bash

   vllm serve ./exported_model --quantization quant_brevitas --enforce-eager

Tensor parallelism is not supported, since quantization parameters are not sharded across GPUs.
To use multiple GPUs or evaluate models that do not fit on a single GPU, see
`Multiple GPUs and Large Models`_.

Zero-Shot Evaluation
====================

The ``brevitas_vllm_eval`` entrypoint evaluates an exported model through vLLM using either
lm-evaluation-harness or LightEval. The latest published harnesses require incompatible vLLM
versions, so install them in separate environments. For lm-evaluation-harness:

.. code-block:: bash

   pip install -e ".[vllm_eval_lm_eval]"

The evaluation backend must be selected explicitly. For lm-evaluation-harness:

.. code-block:: bash

   brevitas_vllm_eval --model ./exported_model --backend lm_eval

In a separate environment, install and run LightEval with:

.. code-block:: bash

   pip install -e ".[vllm_eval_lighteval]"
   brevitas_vllm_eval --model ./exported_model --backend lighteval

Both commands evaluate ARC Challenge, ARC Easy, WinoGrande, and PIQA with zero-shot prompts by
default. With LightEval, the custom tasks defined for ``brevitas_ptq_llm`` (``piqa_lm_eval`` and
``hellaswag_lm_eval``) are available, and ``piqa_lm_eval`` is used by default. Use ``--tasks`` to
override the task list. Task names follow the selected harness; bare LightEval task names are
converted to zero-shot task specifications automatically. For example:

.. code-block:: bash

   brevitas_vllm_eval --model ./exported_model --backend lm_eval \
      --tasks hellaswag piqa --batch-size 64

``--batch-size`` sets the maximum number of sequences processed concurrently by vLLM
(``max_num_seqs``) for both backends. ``--max-new-tokens`` overrides the generation length of every
generative task. ``--add-bos-token`` prepends the BOS token to lm_eval requests and should match
the ``--bos-preprocessing`` setting used with ``brevitas_ptq_llm``; LightEval always adds special
tokens, as it does in ``brevitas_ptq_llm``.

The CLI also exposes ``--dtype``, ``--gpu-memory-utilization``, ``--max-model-length``,
``--limit``, ``--seed``, and ``--output-dir``. As with the LLM quantization entrypoint, these
options can be supplied through a YAML file using ``--config``. For both backends, the numeric
metrics are stored in ``results.json`` within ``--output-dir``. The evaluator always loads the model
with the ``quant_brevitas`` vLLM quantization method, without CUDA graphs and compilation.
Since the export is stored in ``float32``, ``--dtype auto`` runs the model in the preferred half
precision type of the platform; set ``--dtype`` explicitly to match the data type used during
quantization.
The base Brevitas dependency keeps a broad Torch requirement; the vLLM version installed by each
evaluation extra may impose a narrower Torch requirement in that environment.

Thinking Models
---------------

Thinking is disabled by default. Likelihood and multiple-choice tasks always use plain-text
prompts so an empty reasoning block or assistant prefix does not alter continuation scores.
Generative tasks use the model chat template, if defined, and explicitly disable thinking:

.. code-block:: bash

   brevitas_vllm_eval --model ./exported_model --backend lm_eval --tasks gsm8k \
      --thinking disabled

Enable thinking for generative tasks and configure the tags removed before metric computation
with:

.. code-block:: bash

   brevitas_vllm_eval --model ./exported_model --backend lighteval --tasks gsm8k \
      --thinking enabled --reasoning-start-tag '<think>' --reasoning-end-tag '</think>' \
      --max-new-tokens 2048

With lm-evaluation-harness, generative and likelihood tasks must be evaluated in separate
invocations because chat-template and thinking settings apply to the entire evaluation. LightEval
supports mixed task sets by selecting prompt formatting for each request type, provided an
individual task document does not request both generation and likelihood metrics. The reasoning
start and end tags are both used by LightEval. lm-evaluation-harness supports only an end token and
therefore uses ``--reasoning-end-tag`` to remove the reasoning prefix.

Multiple GPUs and Large Models
------------------------------

By default, a single GPU is used, even if more GPUs are visible (e.g., through
``CUDA_VISIBLE_DEVICES``). ``--parallelism`` makes use of all the visible GPUs, and it is supported
by both backends:

* ``data`` runs a replica of the model on each GPU, and each replica evaluates a share of the
  requests. This is the fastest option if the model fits on a single GPU. It relies on Ray, which is
  installed with both evaluation extras. Every replica loads the model, so the CPU memory required to
  load it is multiplied by the number of GPUs.
* ``pipeline`` splits the model layers across the GPUs, for models that do not fit on a single GPU.
  It works for the model architectures that support pipeline parallelism in vLLM, and it scales less
  than linearly with the number of GPUs.

Models that still do not fit can offload part of their weights with ``--cpu-offload-gb X``, which
keeps ``X`` GiB of weights per GPU in CPU memory and streams them to the GPU at every forward pass.
It is supported only by lm-evaluation-harness, since LightEval does not expose this vLLM option.
Evaluation becomes considerably slower, especially for generative tasks, and ``X`` GiB of pinned CPU
memory are required for each GPU.

Tensor parallelism and offloading to disk are not supported. Since quantization is simulated, the
weights take as much memory as the unquantized model in the selected ``--dtype``.

Evaluation in Benchmarks
------------------------

For a single model, quantization and evaluation are two separate commands:

.. code-block:: bash

   brevitas_ptq_llm --model org/model --weight-bit-width 4 --weight-quant-type sym \
     --weight-quant-granularity per_group --weight-group-size 128 \
     --export-target vllm --export-prefix ./exported_model
   brevitas_vllm_eval --config vllm_eval.yaml --model ./exported_model

In benchmark sweeps (``brevitas_examples.llm.benchmark.llm_benchmark`` and ``llm_rand_benchmark``),
every job can instead evaluate its own export through vLLM, by specifying a ``brevitas_vllm_eval``
YAML configuration with ``vllm_eval_config``, together with ``export_target: vllm``:

.. code-block:: yaml

   # vllm_eval.yaml
   backend: lm_eval
   tasks: [arc_challenge, arc_easy, winogrande, piqa]
   dtype: bfloat16
   gpu_memory_utilization: 0.8

.. code-block:: yaml

   # sweep.yaml
   model: [org/model]
   weight_bit_width: [4, 8]
   gptq: [false, true]
   export_target: [vllm]
   vllm_eval_config: [/path/to/vllm_eval.yaml]

The configuration is checked before any job starts. After quantization, each job exports the model
into ``vllm_export`` within its job folder, and ``brevitas_vllm_eval`` runs in a separate process on
the GPUs of the job, storing its outputs in ``vllm_eval``. Its metrics are added to the results with
a ``vllm_`` prefix, so that they appear in ``results.csv`` next to the metrics computed by Brevitas.
To save disk space, the export is deleted after the evaluation, even if it fails, unless
``vllm_eval_keep_export`` is set to ``true``.

The benchmark process keeps part of the GPU memory while vLLM runs, so ``gpu_memory_utilization``
might need to be lowered. The evaluation can use all the GPUs assigned to each job through
``--num-gpus-per-process``, by setting ``parallelism`` in the evaluation configuration.


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
The vLLM-specific code is imported only by the vLLM export and evaluation entrypoints, so vLLM
is not required for other Brevitas workflows.
