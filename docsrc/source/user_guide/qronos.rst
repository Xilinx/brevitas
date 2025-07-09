Post-Training Quantization with Qronos
=================================================================

Qronos is a new post-training quantization (PTQ) algorithm that sequentially rounds 
and updates neural network weights to explicitly address quantization errors that 
have been introduced in both the weights and activations of previous layers. At each
iteration, Qronos first selects the quantized weight that optimally corrects the current
approximation error while holding the remaining weights fixed. It then updates the future
(yet-to-be quantized) weights to optimally compensate for the rounding error. Let's dive
into the Qronos algorithm and how to use it with Brevitas!

.. raw:: html

    <div align="center">
	   <a href="https://arxiv.org/pdf/2505.11695">📄 Paper</a>&nbsp
		<a href="https://github.com/Xilinx/brevitas/blob/dev/src/brevitas/graph/qronos.py">💻 Code</a>
    </div>


.. contents:: Table of Contents
   :local:
   :depth: 3


About the Algorithm
-------------------

PTQ techniques typically aim to solve the layerwise reconstruction problem given by

.. math::

   \operatorname{argmin}_Q \Vert X^T W - X^T Q \Vert

where, for inputs :math:`X` and weights :math:`W`, the goal is to find the quantized 
weights :math:`Q` that minimize the impact of quantization error on the behavior of the 
model.

However, this formulation has no awareness of errors resulting from previously quantized 
layers and/or activation quantization; therefore, algorithms designed to solve the standard 
reconstruction problem (e.g., GPTQ, more recently known as OPTQ [1]) cannot explicitly 
correct for these sources of error.

Qronos considers the “mismatched” reconstruction problem (initially formulated and analyzed 
by Lybrand and Saab [2]), which explicitly addresses these questions via

.. math::

	\operatorname{argmin}_Q \Vert X^T W - \tilde{X}^T Q \Vert

where :math:`\tilde{X}` is the (potentially quantized) inputs from the previously quantized layers.

To solve this problem, Qronos quantizes weights one-by-one by alternating between two steps: 
(1) error correction, where a quantized weight is selected to optimally correct the error; and 
(2) error diffusion, where the remaining unquantized weights are updated to compensate for the 
accumulated rounding error. To do so efficiently, Qronos benefits from the same techniques used 
to scale GPTQ to increasingly large models (e.g., Cholesky decomposition and lazy batch 
updates), and consistently produces quantized models with better accuracy!

🔍 Check out the `paper <https://arxiv.org/pdf/2505.11695>`_ for formalized objective 
functions, derivations, and analyses!


Getting Started
--------------------------------------

Below are the versions used for these results; different versions may yield different results.

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

You can install and build a fork of the ``fast_hadamard_transform`` library with ROCm support 
via:

.. code:: shell

   git clone https://github.com/jeffdaily/fast-hadamard-transform -b rocm
   cd fast-hadamard-transform
   pip install -e .

There is a known issue with ``lighteval`` v0.6.0 (see `#489 
<https://github.com/huggingface/lighteval/issues/489>`_). 
To collect zero-shot results, we use the patched fork:

.. code:: shell

   git clone https://github.com/Giuseppe5/lighteval
   cd lighteval
   pip install .

There is also a known issue with ``transformers==4.51.3`` when also using
``torch=2.4`` (see `#38271 <https://github.com/huggingface/transformers/issues/38271>`_), 
which only impacts QuaRot and SpinQuant. You can install a patched fork via:

.. code:: shell

   git clone https://github.com/i-colbert/transformers -b v4.51.3-patch
   cd transformers
   pip install -e .

Note that you may be able to avoid this issue with later versions of ``torch``, but your 
results may differ from those reported here.

How to Use: Few-Bit LLM Quantization
--------------------------------------

With Brevitas, you can apply the Qronos algorithm to quantize HuggingFace models via
`our LLM entry point <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/llm>`_!

We provide packaged config files in `brevitas_examples/papers/qronos 
<https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/qronos>`_ to enable 
similar experiments described in the paper. The provided configurations specify Llama-3.2-1B, 
but you can specify different Huggingface models in the CLI args. For example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --model=meta-llama/Llama-3.2-3B-Instruct

The BF16 baselines give a WikiText2 perplexity of 8.94 and an average normalized 0-shot 
accuracy (reported as "all_acc_norm" in LightEval) of 59.40% via:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --no-quantize

🧪 Next, we will share our results for weight-only quantization and weight-activation quantization 
for Llama3.2-1B. We encourage you to try more models and formats, and share your results!

Weight-only quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weight-only quantization compresses neural networks by quantizing just the weights (e.g., 
INT4), while keeping activations in full precision (e.g., BF16). It reduces model size and 
memory usage, often with minimal impact on accuracy if one is intentional with calibration.
Here, we will demonstrate how you can use Qronos to calibrate weights quantized to 4 or fewer 
bits.

3 and 4 bit weights
"""""""""""""""""""""""

Below, we summarize the results when quantizing only the weights of Llama-3.2-1B to 3 or 4 
bits. We compare Qronos to GPTQ and GPFQ. We provide round-to-nearest (RTN) as a baseline,
where weights are directly casted to the data format and no calibration is applied.

+--------+------------------------+------------------------+
|        |         3-bit          |         4-bit          |
+--------+-----------+------------+-----------+------------+
|        | Wiki2 (↓) | 0-shot (↑) | Wiki2 (↓) | 0-shot (↑) |
+--------+-----------+------------+-----------+------------+
| RTN    |   2e4     |  32.24     |  18.00    |  48.95     |
+--------+-----------+------------+-----------+------------+
| GPTQ   |  40.50    |  38.15     |  10.44    |  55.39     |
+--------+-----------+------------+-----------+------------+
| GPFQ   |  40.50    |  37.34     |  10.56    |  54.88     |
+--------+-----------+------------+-----------+------------+
| Qronos |**22.00**  |**40.32**   |**10.12**  |**55.87**   |
+--------+-----------+------------+-----------+------------+

You can collect 4-bit weight-only results with the ``lama3-w4-base.yml`` config via:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --qronos

You can instead specify GPTQ or GPFQ by using ``--gptq`` or ``--gpfq`` instead, which are 
mutually exclusive algorithms. You can also specify a different bit width in the CLI args.
For example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --weight-bit-width=3 --qronos

However, we recommend the following config when quantizing to 2 bits or fewer.


2 or 1.58 bit weights
"""""""""""""""""""""""

Quantizing to 2 bits or fewer with minimal degradation requires an intential effort to reduce 
quantization error that arises from different sources. Indeed, the latest innovations in PTQ 
are skewed towards proposing or improving transformations that make weights and/or activations 
more amenable to quantization by limiting the impact of outliers, which is another source of 
quantization error. With Brevitas, you can compose one or more of these transformations with 
Qronos to jointly reduce the impact of outliers while correcting quantization in both weights 
and activations.

The following table summarizes the results of weight-only quantization on Llama-3.2-1B 
when jointly using Hadamard-based incoherence processing (HIP) [3] and weight magnitude 
reduction (MagR)[4] as our quantization transform. We then compare adaptive rounding functions 
when quantizing the model to 1.58-bit (i.e., ternary) or 2-bit weights.

+--------+-----------+------------+-----------+------------+
|        |      1.58-bit          |       2-bit            |
+--------+-----------+------------+-----------+------------+
|        | Wiki2 (↓) | 0-shot (↑) | Wiki2 (↓) | 0-shot (↑) |
+--------+-----------+------------+-----------+------------+
| RTN    |   2e5     |  32.78     |   3e3     |  32.22     |
+--------+-----------+------------+-----------+------------+
| OPTQ   |   3e2     |  33.09     |  25.00    |  38.96     |
+--------+-----------+------------+-----------+------------+
| GPFQ   |   1e2     |  33.21     |  26.25    |  38.73     |
+--------+-----------+------------+-----------+------------+
| Qronos |**39.25**  |**34.11**   |**18.00**  |**42.42**   |
+--------+-----------+------------+-----------+------------+

We provide ``llama3-w2-hip-magr.yml`` as an example, which you can run via:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w2-hip-magr.yml --weight-bit-width=2 --qronos

and you can quantize to 1.58 bits via:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w2-hip-magr.yml --weight-bit-width=2 --weight-narrow-range --qronos

where ``--weight-bit-width=2 --weight-narrow-range`` restricts the
quantization alphabet to :math:`\mathcal{A}=\{-1, 0, 1\}`.


Weight-activation quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Weight-activation quantization constrains both weights and activations to low-precision formats 
(e.g., INT4 or MXFP4), enabling low-precision computations. It also offers memory and compute 
savings, but often requires more careful calibration to maintain accuracy.

QuaRot with INT4 and MXFP4
""""""""""""""""""""""""""""""

QuaRot [3] is a rotation-based quantization method that applies Hadamard transformations to 
neural network weights and activations to remove outliers before quantization, enabling 
accurate low-bit quantization. With Brevitas, you can similarly apply and fuse Hadamard 
rotations then apply Qronos (or other adaptive rounding alorithms). The following table 
summarizes the results of quantizing the weights and activations of Llama-3.2-1B to INT4 or 
MXFP4. We compare Qronos with GPTQ and GPFQ and provide RTN as a baseline.

+--------+-----------+------------+-----------+------------+
|        |       INT4             |           MXFP4        |
+--------+-----------+------------+-----------+------------+
|        | Wiki2 (↓) | 0-shot (↑) | Wiki2 (↓) | 0-shot (↑) |
+--------+-----------+------------+-----------+------------+
| RTN    |  18.00    |  48.31     |  15.38    |  49.53     |
+--------+-----------+------------+-----------+------------+
| OPTQ   |  12.94    |  50.58     |  12.00    |  52.93     |
+--------+-----------+------------+-----------+------------+
| GPFQ   |**12.38**  |**52.73**   |**11.25**  |  53.45     |
+--------+-----------+------------+-----------+------------+
| Qronos |**12.38**  |  51.86     |**11.25**  |**53.71**   |
+--------+-----------+------------+-----------+------------+

To apply weight-activation quantization with Hadamard rotations similar to QuaRot [4], we 
provide ``llama3-w4a4-int-quarot.yml`` and ``llama3-w4a4-mxfp-quarot.yml``. For example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4a4-int-quarot.yml --qronos

Again, using ``--gptq`` or ``--gpfq`` would instead run GPTQ or GPFQ.

SpinQuant with INT4 and MXFP4
"""""""""""""""""""""""""""""""""""

SpinQuant [5] is a more recent rotation-based quantization method that learns rotation matrices 
based on Cayley optimization. With Brevitas, you can similarly learn and fused these rotations, 
then apply Qronos (or other adaptive rounding algorithms). The following table summarizes the 
results of quantizing the weights and activations of Llama-3.2-1B to INT4 or MXFP4 using 
Cayley-optimized rotations. We compare Qronos with GPTQ and GPFQ and provide RTN as a baseline.

+--------+-----------+------------+-----------+------------+
|        |       INT4             |           MXFP4        |
+--------+-----------+------------+-----------+------------+
|        | Wiki2 (↓) | 0-shot (↑) | Wiki2 (↓) | 0-shot (↑) |
+--------+-----------+------------+-----------+------------+
| RTN    |  12.25    |  52.08     |  11.76    |  53.61     |
+--------+-----------+------------+-----------+------------+
| OPTQ   |  12.30    |  53.09     |  11.79    |  53.25     |
+--------+-----------+------------+-----------+------------+
| GPFQ   |  12.28    |  52.85     |  11.35    |  53.22     |
+--------+-----------+------------+-----------+------------+
| Qronos |**11.52**  |**54.00**   |**10.80**  |**54.83**   |
+--------+-----------+------------+-----------+------------+

Unlike the original SpinQuant proposal, which learns rotations after activation quantization 
but before weight quantization, Brevitas learns rotations after quantizing both weights and 
activations. Interestingly, only Qronos is able to improve both perplexity and 0-shot 
performance over RTN.

To apply Cayley-optimized rotations similar to SpinQuant [5], we use 
``llama3-w4a4-int-spinquant.yml`` and ``llama3-w4a4-mxfp-spinquant``. These can be run for 
example:

.. code:: shell

   brevitas_ptq_llm --config=config/llama3-w4a4-int-spinquant.yml --qronos

Again, adding ``--gptq`` or ``--gpfq`` would instead run GPTQ or GPFQ.


GGUF:Q4_0 model export for llama.cpp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can also export the quantized model to several GGUF formats for use with llama.cpp as 
described in our `GGUF export documentation 
<https://xilinx.github.io/brevitas/dev/user_guide/export_gguf.html>`_.

In this example, we export the quantized models to the GGUF:Q4_0 format

.. code:: shell

   brevitas_ptq_llm --config=llama3-gguf-q4_0.yml --qronos

Note that the file "Llama-3.2-1B-1.2B-Q4_0.gguf" will be created in the current directory.

The following table summarizes the results of weight-only quantization of Llama-3.2-1B to 
the GGUF:Q4_0  format, comparing Qronos with GPTQ and GPFQ, where RTN is again provided as a 
baseline.

+--------+-----------+------------+
|        | Wiki2 (↓) | 0-shot (↑) |
+--------+-----------+------------+
| RTN    |  10.44    |  56.81     |
+--------+-----------+------------+
| OPTQ   |   9.50    |  57.96     |
+--------+-----------+------------+
| GPFQ   |   9.50    |**57.99**   |
+--------+-----------+------------+
| Qronos | **9.31**  |  57.88     |
+--------+-----------+------------+


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

Note that this tutorial is not intended to reproduce all the experiments from the original 
paper. To more accurately reproduce experiments from the paper, please see `this 
<https://github.com/i-colbert/brevitas/tree/qronos/src/brevitas_examples/llm>`_ branch.

References
-----------
[1] Frantar, Elias, et al. "OPTQ: Accurate post-training quantization for generative pre-trained transformers." 11th International Conference on Learning Representations. 2023.

[2] Lybrand, Eric, and Rayan Saab. "A greedy algorithm for quantizing neural networks." Journal of Machine Learning Research 22.156 (2021): 1-38.

[3] Ashkboos, Saleh, et al. "QuaRot: Outlier-free 4-bit inference in rotated LLMs." Advances in Neural Information Processing Systems 37 (2024): 100213-100240.

[4] Zhang, Aozhong, et al. "MagR: Weight magnitude reduction for enhancing post-training quantization." arXiv preprint arXiv:2406.00800 (2024).

[5] Liu, Zechun, et al. "SpinQuant: LLM quantization with learned rotations." arXiv preprint arXiv:2405.16406 (2024).

