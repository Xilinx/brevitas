Post-Training Quantization with Qronos
=================================================================

Qronos is a new post-training quantization (PTQ) algorithm that sequentially rounds 
and updates neural network weights to explicitly correct quantization errors in both 
the weights and activations of previous layers while diffusing error into future 
(yet-to-be quantized) weights.

📄 Paper: https://arxiv.org/pdf/2505.11695 
💻 Code: https://github.com/Xilinx/brevitas/blob/dev/src/brevitas/graph/qronos.py 

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
increasingly large models (e.g., Cholesky decomposition and lazy batch updates), and consistently produces 
quantized models with better accuracy!

Check out the `paper <https://arxiv.org/pdf/2505.11695>`_ for formalized objective functions, derivations, 
and analyses.

How to Use: Few-Bit LLM Quantization
--------------------------------------

With Brevitas, you can apply Qronos to quantize models via
`our LLM entry point <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/llm>`_!

We provide packaged configurations in the ``configs/qronos`` folder to enable similar experiments used 
for the paper, and you can specify a Huggingface model via ``--model=meta-llama/Llama-3.2-1B``, for example. 
The provided configurations specify Llama-3.2-1B.

The BF16 baselines give a WikiText2 perplexity of TODO! and an average normalized 0-shot accuracy of TODO! via:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --no-quantize


4-bit weight-only quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following table summarizes the results of weight-only quantization on Llama-3.2-1B 
to 3-bit or 4-bit weights, comparing Qronos with other algorithms like GPTQ, GPFQ, where
round-to-nearest (RTN) is provided as a baseline.

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

You can collect 4-bit weight-only results with ``config/qronos/lama3-w4-base.yml`` via:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml

The provided config runs round-to-nearest (RTN), but you can run GPTQ , GPFQ, or Qronos by 
adding ``--gptq``, ``--gpfq``, or ``--qronos``, respectively, for example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --qronos

You can also specify a different bit width, for example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-w4-base.yml --weight-bit-width=3


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
quantizing the model to 1.58-bit (ternary) or 2-bit weights, comparing Qronos with other
algorithms like GPTQ or GPFQ, where round-to-nearest (RTN) is provided as a baseline.

+--------+--------------------+--------------------+
|        |      1.58-bit      |       2-bit        |
+--------+----------+---------+----------+---------+
|        |  Wiki2   | 0-shot  |  Wiki2   | 0-shot  |
+--------+----------+---------+----------+---------+
| RTN    |          |         |          |         |
+--------+----------+---------+----------+---------+
| OPTQ   |          |         |          |         |
+--------+----------+---------+----------+---------+
| GPFQ   |          |         |          |         |
+--------+----------+---------+----------+---------+
| Qronos |          |         |          |         |
+--------+----------+---------+----------+---------+

We provide ``config/llama3-w2-hip-magr.yml`` as an example, which you can run via:

.. code:: shell

   brevitas_ptq_llm --config=config/llama3-w2-hip-magr.yml --weight-bit-width=2 --qronos

Note that you can quantize to 1.58 bits via:

.. code:: shell

   brevitas_ptq_llm --config=config/llama3-w2-hip-magr.yml --weight-bit-width=2 --weight-narrow-range

where ``--weight-bit-width=2 --weight-narrow-range`` restricts the
quantization alphabet to :math:`\mathcal{A}=\{-1, 0, 1\}`.


4-bit weight-activation quantization with QuaRot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------+--------------------+--------------------+
|        |       INT4         |       MXFP4        |
+--------+----------+---------+----------+---------+
|        |  Wiki2   | 0-shot  |  Wiki2   | 0-shot  |
+--------+----------+---------+----------+---------+
| RTN    |          |         |          |         |
+--------+----------+---------+----------+---------+
| OPTQ   |          |         |          |         |
+--------+----------+---------+----------+---------+
| GPFQ   |          |         |          |         |
+--------+----------+---------+----------+---------+
| Qronos |          |         |          |         |
+--------+----------+---------+----------+---------+


To apply weight-activation quantization with Hadamard rotations, similar to what is proposed for QuaRot [4], we 
provide ``config/llama3-w4a4-quarot.yml``. Similarly, to apply Cayley-optimized rotations similar to what is proposed 
for SpinQuant [5], we use ``config/llama3-w4a4-spinquant.yml``. These can be run for example:

.. code:: shell

   brevitas_ptq_llm --config=config/llama3-w4a4-quarot.yml --qronos

Again, running ``--gptq`` or ``--gpfq`` would instead GPTQ or GPFQ.

4-bit weight-activation quantization with SpinQuant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------+--------------------+--------------------+
|        |       INT4         |       MXFP4        |
+--------+----------+---------+----------+---------+
|        |  Wiki2   | 0-shot  |  Wiki2   | 0-shot  |
+--------+----------+---------+----------+---------+
| RTN    |          |         |          |         |
+--------+----------+---------+----------+---------+
| OPTQ   |          |         |          |         |
+--------+----------+---------+----------+---------+
| GPFQ   |          |         |          |         |
+--------+----------+---------+----------+---------+
| Qronos |          |         |          |         |
+--------+----------+---------+----------+---------+

How to Use: Few-Bit ConvNet Quantization
-------------------------------------------

With Brevitas, one can also apply Qronos to quantize models via  `our TorchVision entry point <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/imagenet_classification/ptq>`_!

Similar to our LLM entry point, several techniques can be composed. For example, to run Qronos via the TorchVision entry point on GPU 0:

.. code:: shell

   brevitas_ptq_imagenet_val --calibration-dir=/path/to/imagenet/calibration/folder --validation-dir=/path/to/imagenet/validation/folder --gpu=0 --model-name=resnet50 --qronos

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

Please use `this branch <https://github.com/i-colbert/brevitas/tree/qronos/src/brevitas_examples/llm>`_ to reproduce the experiments used for the paper.

References
-----------
[1] Frantar, Elias, et al. "OPTQ: Accurate post-training quantization for generative pre-trained transformers." 11th International Conference on Learning Representations. 2023.

[2] Lybrand, Eric, and Rayan Saab. "A greedy algorithm for quantizing neural networks." Journal of Machine Learning Research 22.156 (2021): 1-38.

[3] Zhang, Aozhong, et al. "MagR: Weight magnitude reduction for enhancing post-training quantization." arXiv preprint arXiv:2406.00800 (2024).

[4] Ashkboos, Saleh, et al. "QuaRot: Outlier-free 4-bit inference in rotated LLMs." Advances in Neural Information Processing Systems 37 (2024): 100213-100240.

[5] Liu, Zechun, et al. "SpinQuant: LLM quantization with learned rotations." arXiv preprint arXiv:2405.16406 (2024).

