Post-Training Quantization with Model Expansion
=================================================================

Post-training model expansion is a new strategy to improve quantization quality by 
modestly increasing model size while still significantly reducing overall parameter 
volume. Rather than only reducing bit widths, this approach strategically expands 
certain layers post-training to bridge the gap between full-precision and quantized 
models. When quantizing Llama3 1B to 4-bit weights and activations, this technique 
reduces the gap to full-precision perplexity by an average of 9% compared to QuaRot 
and SpinQuant with only 5% more parameters—still achieving a 3.8x volume reduction 
relative to BF16.

.. raw:: html

    <div align="center">
	   <a href="https://arxiv.org/abs/2503.17513">📄 Paper</a>&nbsp
		<a href="https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/expansion">💻 Examples</a>
    </div>


.. contents:: Table of Contents
   :local:
   :depth: 3


About the Algorithm
-------------------

Traditional quantization focuses on reducing parameter volume primarily through bit width 
reduction. However, in real-world deployments where maintaining model quality is critical, 
a modest 5-10% increase in model size is often an acceptable trade-off. For example, when 
deploying on hardware accelerators restricted to power-of-two bit widths (e.g., GPUs), 
failing to meet accuracy requirements at 4-bit necessitates reverting to 8-bit—a significant 
jump in parameter volume.

Post-training model expansion offers a complementary dimension in the quantization design 
space. The key insight is to expand Hadamard matrices used during online incoherence 
processing. Recent quantization methods like QuaRot [1] and SpinQuant [2] already insert 
Hadamard rotations into the compute graph to reduce outliers. This work extends that idea 
by using expanded Hadamard matrices that increase certain layer dimensions post-training.

The Expansion Mechanism
~~~~~~~~~~~~~~~~~~~~~~~

Incoherence processing exploits rotation invariance in linear layers. Given orthogonal 
rotation matrix :math:`R`, we have:

.. math::

   XW = (XR)(R^TW) = XRR^{-1}W = XW

where :math:`X` are inputs and :math:`W` are weights. The rotation :math:`R` effectively 
rotates the latent space to amortize outliers before quantizing :math:`XR` and :math:`R^TW`.

For expansion, instead of using an :math:`N \times N` Hadamard matrix :math:`H`, we generate 
an :math:`M \times M` matrix where :math:`M > N`, then select only the first :math:`N` rows 
to yield :math:`\hat{H} \in \mathbb{R}^{N \times M}`. This expanded matrix increases the 
number of input channels from :math:`N` to :math:`M`. The left inverse is still computed 
through transposition since :math:`\hat{H}\hat{H}^T = I`, preserving computational efficiency.

Why Does It Work?
~~~~~~~~~~~~~~~~~

Expansion provides two theoretical benefits:

**1. Increased Nullspace:** By the rank-nullity theorem, expanding from :math:`N` to :math:`M` 
dimensions strictly increases the nullspace by :math:`M - N`. This provides more opportunity 
to hide quantization error during calibration, since we seek quantized weights :math:`Q(w)` 
where :math:`w - Q(w)` lies in the nullspace of the input activations.

**2. Reduced Error Bounds:** For GPTQ-based quantization, expansion reduces the supremum of 
reconstruction error by a factor of :math:`\sqrt{N/M}`. While this doesn't guarantee universal 
improvement, it establishes a theoretical rate of decay for worst-case error under ideal conditions.

🔍 Check out the `paper <https://arxiv.org/abs/2503.17513>`_ for complete mathematical 
derivations and formal proofs!


Getting Started
--------------------------------------

Install Brevitas and the required dependencies:

.. code:: shell
   
   pip install brevitas[llm, export] lighteval


How to Use: Expanding Models Post-Training
--------------------------------------------

With Brevitas, you can apply post-training model expansion to quantize HuggingFace models 
via `our LLM entry point <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/llm>`_!

The technique works by expanding specific Hadamard rotation matrices in the compute graph. 
Two main expansion strategies are available:

1. **Expanding R4** (down projection only): Lower overhead, focused improvement
2. **Expanding R2 and R4** (attention output + down projection): Broader improvement

These expansions can be combined with either QuaRot-style Hadamard rotations or 
SpinQuant-style Cayley-optimized rotations. Let's explore the results!


Expanding Attention and Down Projection (R2 + R4)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Expanding both the output projection in attention (R2) and the down projection (R4) 
provides the best overall improvements. This strategy increases model size by approximately 
14-17% while delivering substantial perplexity gains.

INT4 Weight-Activation Quantization
""""""""""""""""""""""""""""""""""""

The following table shows results for 4-bit weight and activation quantization with expanded 
models. We use per-channel symmetric weight quantization and per-token asymmetric activation 
quantization.

+-------------------+-----------+------------+-----------+------------+-----------+------------+
|                   |         Llama 3.2 1B   |         Llama 3.2 3B   |        Llama 3.1 8B    |
+-------------------+-----------+------------+-----------+------------+-----------+------------+
|                   | Wiki2 (↓) | 0-shot (↑) | Wiki2 (↓) | 0-shot (↑) | Wiki2 (↓) | 0-shot (↑) |
+-------------------+-----------+------------+-----------+------------+-----------+------------+
| Float             | 8.94      | 54.08      | 7.16      | 62.19      | 5.91      | 67.95      |
+-------------------+-----------+------------+-----------+------------+-----------+------------+
| Base              | 12.94     | 46.67      | 9.06      | 57.15      | 7.34      | 62.41      |
+-------------------+-----------+------------+-----------+------------+-----------+------------+
|  +14%/+16%/+17%   |**12.19**  |**47.40**   |**8.75**   |**57.10**   |**7.06**   |**63.30**   |
+-------------------+-----------+------------+-----------+------------+-----------+------------+

The expanded models consistently improve both perplexity and zero-shot accuracy across 
all model sizes.

When combined with SpinQuant's Cayley-optimized rotations, even greater improvements 
are possible:

+--------+-----------+------------+
|        | Llama 3.2 1B           |
+--------+-----------+------------+
|        | Wiki2 (↓) | 0-shot (↑) |
+--------+-----------+------------+
| Float  | 8.94      | 54.08      |
+--------+-----------+------------+
| Base   | 12.11     | 48.46      |
+--------+-----------+------------+
| +14%   |**11.40**  |**48.83**   |
+--------+-----------+------------+

MXFP4 Weight-Activation Quantization
"""""""""""""""""""""""""""""""""""""

The expansion technique also works with the OCP MXFP4 datatype, which uses E8M0 microscaling 
for groups of 32 elements. This format offers higher precision through finer-grained scaling 
factors.

+------------------+-----------+------------+-----------+------------+-----------+------------+
|                  |    Llama 3.2 1B        |         Llama 3.2 3B   |    Llama 3.1 8B        |
+------------------+-----------+------------+-----------+------------+-----------+------------+
|                  | Wiki2 (↓) | 0-shot (↑) | Wiki2 (↓) | 0-shot (↑) | Wiki2 (↓) | 0-shot (↑) |
+------------------+-----------+------------+-----------+------------+-----------+------------+
| Float            | 8.94      | 54.08      | 7.16      | 62.19      | 5.91      | 67.95      |
+------------------+-----------+------------+-----------+------------+-----------+------------+
| Base             | 11.81     | 48.03      | 8.94      | 54.46      | 7.10      | 63.54      |
+------------------+-----------+------------+-----------+------------+-----------+------------+
| +14%/+16%/+17%   |**11.44**  |**48.73**   |**8.63**   |**58.50**   |**6.94**   |**64.30**   |
+------------------+-----------+------------+-----------+------------+-----------+------------+

With SpinQuant optimization:

+--------+-----------+------------+
|        | Llama 3.2 1B           |
+--------+-----------+------------+
|        | Wiki2 (↓) | 0-shot (↑) |
+--------+-----------+------------+
| Float  | 8.94      | 54.08      |
+--------+-----------+------------+
| Base   | 11.59     | 49.33      |
+--------+-----------+------------+
| +14%   |**11.34**  |**50.31**   |
+--------+-----------+------------+

Expansion provides consistent benefits across both INT4 and MXFP4 formats, demonstrating 
the general applicability of this technique.


Understanding the Quality-Volume Tradeoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Post-training model expansion introduces a new dimension to the quantization design space. 
Instead of only adjusting bit widths, we can now trade modest size increases for better 
quality while maintaining significant volume reductions.

For a fixed memory constraint (measured as model volume = size × bit width), expanding 
quantized models can provide better quality than smaller models at higher precision. 
Notably, the paper shows that quantized Llama 3.2 3B and 8B models with expansion can 
outperform smaller models in BF16 on zero-shot tasks while using less total volume.

This suggests that when deploying under memory constraints, the optimal solution may 
involve a larger quantized model rather than a smaller full-precision model.


Example Commands
~~~~~~~~~~~~~~~~

To quantize Llama3 models with expansion using Brevitas, you can use the LLM quantization 
entry point with expansion-enabled configurations. You can modify existing configuration 
files to enable Hadamard expansion by setting the appropriate expansion parameters for 
the target rotation matrices (R2, R4, etc.).

For examples, it is possible to reproduce the results in our paper following the benchmark
yamls in `brevitas_examples/papers/expansion`.


.. code:: shell

   python benchmark.py --config quarot_star.yaml


Citation
--------

::

   @article{franco2025posttraining,
         title={Improving Quantization with Post-Training Model Expansion}, 
         author={Giuseppe Franco and Pablo Monteagudo-Lago and Ian Colbert and Nicholas Fraser and Michaela Blott},
         year={2025},
         eprint={2503.17513},
         archivePrefix={arXiv},
         primaryClass={cs.LG},
         url={https://arxiv.org/abs/2503.17513}, 
   }

Note that this tutorial provides an overview of the technique and representative results. 
To reproduce all experiments from the paper exactly, please refer to the configurations 
and instructions in the paper's supplementary materials.


References
-----------

[1] Ashkboos, Saleh, et al. "QuaRot: Outlier-free 4-bit inference in rotated LLMs." arXiv preprint arXiv:2404.00456 (2024).

[2] Liu, Zechun, et al. "SpinQuant: LLM quantization with learned rotations." arXiv preprint arXiv:2405.16406 (2024).

[3] Frantar, Elias, et al. "GPTQ: Accurate post-training quantization for generative pre-trained transformers." arXiv preprint arXiv:2210.17323 (2022).

[4] Zhao, Ritchie, et al. "Improving neural network quantization without retraining using outlier channel splitting." International conference on machine learning. PMLR, 2019.

[5] Yu, Mengxia, et al. "The super weight in large language models." arXiv preprint arXiv:2411.07191 (2024).

[6] Darvish Rouhani, Bita, et al. "Microscaling data formats for deep learning." arXiv preprint arXiv:2310.10537 (2023).
