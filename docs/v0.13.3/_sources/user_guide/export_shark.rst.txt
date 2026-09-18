====================
Shark-AI Export
====================


`Shark-AI <https://github.com/nod-ai/shark-ai>`_ is a new AMD open-source compilation tool, 
that can target the latest AMD hardware and most recent state-of-the-art 
networks, with options for both quantized and full precision configurations.

Within Brevitas, we are closely collaborating with the team behind Shark-AI to provide an easy to use
quantization flow for all the latest and greatest SOTA models, that can then be deployed through
Shark-AI.

Current preliminary support is focused on LLM, but we are already planning the next steps 
of this integration to enable support to other types of architectures.

The current recommended way to export a model to Shark-AI is through our LLM entrypoint.

LLM Entrypoint
==============

Brevitas' LLM entrypoint allows the user to load, quantize, test, and export many of the LLM available on 
HuggingFace, by simply passing a series of command line arguments that can control, among other things:

* Weights and activations bit width
* Weights and activation quantization format (int vs float, asym vs sym, etc.)
* PTQ algorithms to apply and their options
* and much more...

Exporting a HuggingFace model for Shark-AI can be as easy as running the following:

.. code-block:: bash

    brevitas_ptq_llm --model org/model --input-bit-width 8 --weight-bit-width 8 --input-quant-format float_fnuz_e4m3 --weight-quant-format float_fnuz_e4m3 --input-quant-granularity per_tensor --weight-quant-granularity per_tensor --act-calibration --input-quant-type sym --export-target shark --eval --export-prefix path/to/folder

In particular, this quantization configuration corresponds to the following:

* All linear layers (except for the last one) weights and activations quantized to FP8
* Per tensor scale factors for both

Many more options are available through the entrypoint, such as the possibility to quantize KV cache 
or the entire attention operator.
For example, to also perform FP8 attention quantization, simply add:

.. code-block:: bash

    --quant-sdpa eager --attn-quant-config qkv

Other options that can be specified by the user include the possibility to change the scale factors granularity or decide 
what PTQ algorithms to use.
For a more exhaustive list of options, check our `README <https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/llm/README.md>`_.

Currently, dynamic activation quantization is not supported in Shark-AI.
Furthemore, although Shark-AI has supports for per-group scale factors, it is not easy to target 
that configuration starting from Brevitas' entrypoint.

Both Integer and FP8 quantization are supported for accelerated execution.

Not all the available PTQ algorithms can be used when exporting to Shark-AI.
In particular, the following are not supported:

* QuaRot/SpinQuant with standalone Hadamard rotations
* SVDQuant
* Learned Round


This still leaves a lot of options available, such as: 

* MagR
* Activation Equalization (i.e., SmoothQuant)
* AWQ
* QuaRot/SpinQuant with only fused Hadamard rotations
* GPTQ/GPFQ
* Qronos
* Bias Correction


Compilation step
=================


Once the export process is completed, a `dataset.irpa` file will be create in the folder specified as
argument to `export-prefix`.

To use this file through Shark-AI, make sure to follow the `installation instructions <https://github.com/nod-ai/shark-ai/blob/main/docs/user_guide.md>`_:

Before running your accelerated LLM, there are two steps required through Shark-AI, export and compilation.


The export command can be run as follow:

.. code-block:: bash

    python -m sharktank.examples.export_paged_llm_v1 --irpa-file=path/to/folder/dataset.irpa --output-mlir=model.mlir --output-config=config.json --bs-prefill=16 --bs-decode=16 --activation-dtype=float32 --attention-dtype=float8_e4m3fnuz --attention-kernel=sharktank --kv-cache-dtype=float8_e4m3fnuz --use-hf --use-attention-mask

This command assumes that also the attention part of the network was quantized to FP8. If that is not 
the case, replace the attention dtype flag with:

.. code-block:: bash

    --attention-dtype=float32

After this step, you should have a new file called `model.mlir`, which is what we need for the compilation phase,
which can be done as follows:

.. code-block:: bash

    iree-compile ./model.mlir --iree-hal-target-device=hip --iree-hip-target=gfx942 -o model.vmfb --iree-opt-level=O3 --iree-hal-indirect-command-buffers=true  --iree-stream-resource-memory-model=discrete  --iree-hal-memoization=true

This command specifically targets MI300X,but other options are available. 


For more information about the various flags in the export and compilation command, please reach out directly
to Shark-AI.


Once these commands have completed succesfully, the `model.vmfb` file can be used for accelerated inference of your model.

For example, to run the benchmark, run the following:

.. code-block:: bash

   iree-benchmark-module --hip_use_streams=true --benchmark_repetitions=5 --parameters="model=path/to/dataset.irpa" --device='hip://1'  --iree-hip-target=gfx942 --module=model.vmfb --function=prefill_bs16 --input=16x1024xsi64 --input=16xsi64 --input=16x32xsi64 --input=4096x2097152xf8E4M3FNUZ

For more information and options, such as the possibility to start the Shortfin LLM server, check the
tutorial avaialble in the `Shark-AI repository <https://github.com/nod-ai/shark-ai/blob/main/docs/shortfin/llm/user/llama_serving.md>`_


This is still an experimental flow and lots of changes and improvements will be made in the future, 
including the possibility of breaking changes.


Next steps
=================

Currently, Brevitas entrypoint allows to quantize many commonly used LLM directly from HuggingFace,
applying the user desired quantization algorithms and then exporting them in a format that Shark-AI 
is able to consume.

On the other hand, not all model families have been tested with Shark-AI. Moreover, the Shark-AI repository 
redefines some models from scratch to ensure the best compatibility and performance when deploying on AMD 
hardware.
In this context, export from HuggingFace to Shark-AI might cause slightly mismatches and inaccuracies.

For this reason, we are working to quantize directly the models defined within Shark-AI.
This flow will allow to insert quantization within a Shark model, apply all the quantization algorithms, 
and then swap back quantized layers with the original versions, before proceeding to the `irpa` export, which will 
be natively handled by the Shark model itself.


If you have further questions, please feel free to reach open an issue either in Brevitas or Shark-AI
and we will do our best to support you.

