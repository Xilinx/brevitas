Post-Training Quantization with Qronos
=================================================================

Qronos is a new post-training quantization (PTQ) algorithm that sequentially rounds 
and updates neural network weights to explicitly correct quantization errors in both 
the weights and activations of previous layers while diffusing error into future 
(yet-to-be quantized) weights.

📄 `Paper <https://arxiv.org/pdf/2505.11695>`_ 
💻 `Code <https://github.com/Xilinx/brevitas/blob/dev/src/brevitas/graph/qronos.py>`_

.. contents:: Table of Contents
   :local:
   :depth: 2

Let’s dive into the Qronos algorithm and how to use it with Brevitas!

About the Algorithm
-------------------

PTQ techniques typically aim to solve the layerwise reconstruction problem given by

:math:`\operatorname{argmin}_Q \Vert X^T W - X^T Q \Vert`

where, for inputs :math:`X` and weights :math:`W`, the goal is to find the quantized 
weights :math:`Q` that minimize the impact of quantization error on the behavior of the 
model.

However, this formulation has no awareness of error resulting from previously quantized 
layers and/or activation quantization; therefore, algorithms derived to solve the standard 
reconstruction problem (e.g., GPTQ, more recently known as OPTQ [1]) cannot explicitly 
correct for these sources of error in the activations.

Qronos considers the “mismatched” reconstruction problem (initially formulated and analyzed 
by Lybrand and Saab [2]), which explicitly addresses these questions via
:math:`\operatorname{argmin}_Q \Vert X^T W - \tilde{X}^T Q \Vert`
where :math:`\tilde{X}` is the (potentially quantized) inputs from the previously quantized layers.

To solve this problem, Qronos quantizes weights one-by-one by alternating between two steps: (1) 
error correction, where a quantized weight is selected to optimally correct the error; and (2) error 
diffusion, where the remaining unquantized weights are updated to compensate for the accumulated 
rounding error. To do so efficiently, Qronos benefits from the same techniques used to scale GPTQ to 
increasingly large models (e.g., Cholesky decomposition and lazy batch updates), and consistently 
produces quantized models with better accuracy!

Check out the `paper <https://arxiv.org/pdf/2505.11695>`_ for formalized objective functions, 
derivations, and analyses.


Getting Started
--------------------------------------

Below are versions used for this work.

- ``python==3.12``
- ``torch==2.4.0+rocm6.1``
- ``datasets==3.2.0``
- ``optimum==1.24.0``
- ``accelerate==1.3.0``
- ``transformers==4.51.3`` (custom fork, see below)
- ``fast_hadamard-transform==1.0.4`` (custom fork, see below)
- ``lighteval==0.6.0`` (custom fork, see below)

You can install PyTorch for ROCm 6.1 via:

.. code:: shell

   pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

You can install and build a fork of the ``fast_hadamard_transform`` library with ROCm support via:

.. code:: shell

   git clone https://github.com/jeffdaily/fast-hadamard-transform -b rocm
   cd fast-hadamard-transform
   pip install -e .

There is a known issue with ``lighteval`` v0.6.0 (see `#489 <https://github.com/huggingface/lighteval/issues/489>`_). 
To collect zero-shot results, we use the patched fork:

.. code:: shell

   git clone https://github.com/Giuseppe5/lighteval
   cd lighteval
   pip install .

There is also a known issue with ``transformers`` v4.51.3 (see `#38271 <https://github.com/huggingface/transformers/issues/38271>`_). 
To use QuaRot and SpinQuant here, we use the patched fork:

.. code:: shell

   git clone https://github.com/i-colbert/transformers -b v4.51.3-patch
   cd transformers
   pip install -e .


How to Use: Few-Bit LLM Quantization
--------------------------------------

With Brevitas, you can apply the Qronos algorithm to quantize HuggingFace models via
`our LLM entry point <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/llm>`_!

We provide packaged config files in the ``configs/qronos`` folder to enable similar experiments used 
for the paper. The provided configurations specify Llama-3.2-1B.

The BF16 baselines give a WikiText2 perplexity of 8.94 and an average normalized 0-shot accuracy 
(or "all_acc_norm" from LightEval) of 59.40% via:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --no-quantize

Note that you can specify different Huggingface models by adding it to the CLI args. For example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --model=meta-llama/Llama-3.2-3B-Instruct


4-bit weight-only quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table summarizes the results of weight-only quantization on Llama-3.2-1B 
to 3-bit or 4-bit weights, comparing Qronos with GPTQ and GPFQ, where round-to-nearest 
(RTN) is provided as a baseline.

+--------+--------------------+--------------------+
|        |       3-bit        |       4-bit        |
+--------+----------+---------+----------+---------+
|        |  Wiki2   | 0-shot  |  Wiki2   | 0-shot  |
+--------+----------+---------+----------+---------+
| RTN    |   2e4    |  32.24  |  18.00   |  48.95  |
+--------+----------+---------+----------+---------+
| GPTQ   |  40.50   |  38.15  |  10.44   |  55.39  |
+--------+----------+---------+----------+---------+
| GPFQ   |  40.50   |  37.34  |  10.56   |  54.88  |
+--------+----------+---------+----------+---------+
| Qronos |  22.00   |  40.32  |  10.12   |  55.87  |
+--------+----------+---------+----------+---------+

You can collect 4-bit weight-only results with the ``config/qronos/lama3-w4-base.yml`` config 
via:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml

The provided config runs RTN, but you can run GPTQ , GPFQ, or Qronos by 
adding ``--gptq``, ``--gpfq``, or ``--qronos``, respectively, for example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --qronos

You can also specify a different bit width, for example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --weight-bit-width=3

However, we recommend the following config when quantizing to 2 bits or fewer.


2-bit and 1.58-bit weight-only quantization 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The latest innovations in PTQ are skewed towards proposing or improving transformations 
that make weights and/or activations more amenable to quantization. These studies often 
focus on round-to-nearest (RTN), but more recent studies explore their interaction with 
adaptive rounding algorithms [3,4,5].  With Brevitas, you can compose one or more of these 
transformations with adaptive rounding algorithms like Qronos, GPTQ, or GPFQ.

The following table summarizes the results of weight-only quantization on Llama-3.2-1B 
when jointly using Hadamard-based incoherence processing (HIP) and weight magnitude reduction 
(MagR) as our quantization transform. We then compare adaptive rounding functions when
quantizing the model to 1.58-bit (i.e., ternary) or 2-bit weights.

+--------+--------------------+--------------------+
|        |      1.58-bit      |       2-bit        |
+--------+----------+---------+----------+---------+
|        |  Wiki2   | 0-shot  |  Wiki2   | 0-shot  |
+--------+----------+---------+----------+---------+
| RTN    |   2e5    |  32.78  |   3e3    |  32.22  |
+--------+----------+---------+----------+---------+
| OPTQ   |   3e2    |  33.09  |  25.00   |  38.96  |
+--------+----------+---------+----------+---------+
| GPFQ   |   1e2    |  33.21  |  26.25   |  38.73  |
+--------+----------+---------+----------+---------+
| Qronos |  39.25   |  34.11  |  18.00   |  42.42  |
+--------+----------+---------+----------+---------+

We provide ``config/llama3-w2-hip-magr.yml`` as an example, which you can run via:

.. code:: shell

   brevitas_ptq_llm --config=config/llama3-w2-hip-magr.yml --weight-bit-width=2 --qronos

and you can quantize to 1.58 bits via:

.. code:: shell

   brevitas_ptq_llm --config=config/llama3-w2-hip-magr.yml --weight-bit-width=2 --weight-narrow-range

where ``--weight-bit-width=2 --weight-narrow-range`` restricts the
quantization alphabet to :math:`\mathcal{A}=\{-1, 0, 1\}`.


4-bit weight-activation quantization with QuaRot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table summarizes the results of weight-activation quantization of
Llama-3.2-1B to INT4 or MXFP4 data formats using Hadamard-based incoherence processing
similar to what is proposed for QuaRot [4]. We compare Qronos with GPTQ and GPFQ, 
where round-to-nearest (RTN) is provided as a baseline.

+--------+--------------------+--------------------+
|        |       INT4         |       MXFP4        |
+--------+----------+---------+----------+---------+
|        |  Wiki2   | 0-shot  |  Wiki2   | 0-shot  |
+--------+----------+---------+----------+---------+
| RTN    |  18.00   |  48.31  |  15.38   |  49.53  |
+--------+----------+---------+----------+---------+
| OPTQ   |  12.94   |  50.58  |  12.00   |  52.93  |
+--------+----------+---------+----------+---------+
| GPFQ   |  12.38   |  52.73  |  11.25   |  53.45  |
+--------+----------+---------+----------+---------+
| Qronos |  12.38   |  51.86  |  11.25   |  53.71  |
+--------+----------+---------+----------+---------+

To apply weight-activation quantization with Hadamard rotations, similar to what is proposed for 
QuaRot [4], we provide ``config/llama3-w4a4-quarot.yml`` and ``config/llama3-w4a4-mxfp-quarot.yml``. 

.. code:: shell

   brevitas_ptq_llm --config=config/llama3-w4a4-quarot.yml --qronos

Again, running ``--gptq`` or ``--gpfq`` would instead GPTQ or GPFQ.

4-bit weight-activation quantization with SpinQuant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table summarizes the results of weight-activation quantization of
Llama-3.2-1B to INT4 or MXFP4 data formats using Hadamard-based incoherence processing
similar to what is proposed for QuaRot [4]. We compare Qronos with GPTQ and GPFQ, 
where round-to-nearest (RTN) is provided as a baseline.

+--------+--------------------+--------------------+
|        |       INT4         |       MXFP4        |
+--------+----------+---------+----------+---------+
|        |  Wiki2   | 0-shot  |  Wiki2   | 0-shot  |
+--------+----------+---------+----------+---------+
| RTN    |  12.25   |  52.08  |  11.76   |  53.61  |
+--------+----------+---------+----------+---------+
| OPTQ   |  12.30   |  53.09  |  11.79   |  53.25  |
+--------+----------+---------+----------+---------+
| GPFQ   |  12.28   |  52.85  |  11.35   |  53.22  |
+--------+----------+---------+----------+---------+
| Qronos |  11.52   |  54.00  |  10.80   |  54.83  |
+--------+----------+---------+----------+---------+

Similarly, to apply Cayley-optimized rotations similar to what is proposed for SpinQuant [5], we 
use ``config/llama3-w4a4-spinquant.yml`` and ``config/llama3-w4a4-mxfp-spinquant``. These can be 
run for example:

.. code:: shell

   brevitas_ptq_llm --config=config/llama3-w4a4-spinquant.yml --qronos

Again, running ``--gptq`` or ``--gpfq`` would instead GPTQ or GPFQ.

GGUF:Q4_0 model export for llama.cpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also export the quantized model to the GGUF formats for use with llama.cpp as 
described in `GGUF Export <https://xilinx.github.io/brevitas/dev/user_guide/export_gguf.html>`_.

In this example, we export the quantized models to the GGUF:Q4_0 format

.. code:: shell

   brevitas_ptq_llm --config=config/llama3-gguf-q4_0.yml --qronos

Note that the file "Llama-3.2-1B-1.2B-Q4_0.gguf" will be created in the current directory.

The following table summarizes the results of weight-only quantization of Llama-3.2-1B to 
the GGUF:Q4_0  format, comparing Qronos with GPTQ and GPFQ, where round-to-nearest (RTN) 
is provided as a baseline.

+--------+----------+---------+
|        |  Wiki2   | 0-shot  |
+--------+----------+---------+
| RTN    |  10.44   |  56.81  |
+--------+----------+---------+
| OPTQ   |   9.50   |  57.96  |
+--------+----------+---------+
| GPFQ   |   9.50   |  57.99  |
+--------+----------+---------+
| Qronos |   9.31   |  57.88  |
+--------+----------+---------+


How to Use: Few-Bit ConvNet Quantization
-------------------------------------------

With Brevitas, one can also apply Qronos to quantize models via  `our TorchVision entry point <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/imagenet_classification/ptq>`_!

For example, to run Qronos via the TorchVision entry point on GPU 0:

.. code:: shell

   brevitas_ptq_imagenet_val --calibration-dir=/path/to/imagenet/calibration/folder --validation-dir=/path/to/imagenet/validation/folder --gpu=0 --model-name=resnet50 --qronos

The following table summarizes the results of weight-activation quantization on MobileNetV2 and ResNet50
to 4-bit weights with either 4-bit or 8-bit activations (W4A4 or W4A8, respectively). We compare Qronos with 
GPTQ and GPFQ, where round-to-nearest (RTN) is provided as a baseline.

+--------+--------------------+--------------------+
|        |    mobilenet_v2    |      resnet50      |
+--------+----------+---------+----------+---------+
|        |   W4A4   |  W4A8   |  W4A4    |  W4A8   |
+--------+----------+---------+----------+---------+
| RTN    |          |         |          |         |
+--------+----------+---------+----------+---------+
| OPTQ   |          |         |          |         |
+--------+----------+---------+----------+---------+
| GPFQ   |          |         |          |         |
+--------+----------+---------+----------+---------+
| Qronos |          |         |          |         |
+--------+----------+---------+----------+---------+


Citation
--------

::

   @article{zhang2025qronos,
         title={Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization}, 
         author={Shihao Zhang and Haoyu Zhang and Ian Colbert and Rayan Saab},
         year={2025},
         eprint={2505.11695},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2505.11695}, 
   }

Note that this tutorial is not intended to reproduce all the experiments from the original paper. To do so, 
please see `this <https://github.com/i-colbert/brevitas/tree/qronos/src/brevitas_examples/llm>`_ branch.

References
-----------
[1] Frantar, Elias, et al. "OPTQ: Accurate post-training quantization for generative pre-trained transformers." 11th International Conference on Learning Representations. 2023.

[2] Lybrand, Eric, and Rayan Saab. "A greedy algorithm for quantizing neural networks." Journal of Machine Learning Research 22.156 (2021): 1-38.

[3] Zhang, Aozhong, et al. "MagR: Weight magnitude reduction for enhancing post-training quantization." arXiv preprint arXiv:2406.00800 (2024).

[4] Ashkboos, Saleh, et al. "QuaRot: Outlier-free 4-bit inference in rotated LLMs." Advances in Neural Information Processing Systems 37 (2024): 100213-100240.

[5] Liu, Zechun, et al. "SpinQuant: LLM quantization with learned rotations." arXiv preprint arXiv:2405.16406 (2024).

