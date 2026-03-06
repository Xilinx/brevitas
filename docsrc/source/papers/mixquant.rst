Post-Training Quantization with MixQuant
=================================================================

MixQuant is a post-training quantization (PTQ) technique that improves the outlier suppression capabilities of block 
Hadamard rotations. Prior to rotation, MixQuant inserts calibrated permutations to redistribute activation mass within
permutation-equivariant regions of the graph. The permutations are calibrated once offline using activation statistics 
and then merged into surrounding weight tensors before deployment so they do not incur additional inference overhead. 
Let's dive into MixQuant and how to use it with Brevitas!

.. raw:: html

    <div align="center">
        <a href="https://arxiv.org/abs/2601.22347">📄 Paper</a>&nbsp
        <a href="https://github.com/Xilinx/brevitas/pull/1448">💻 Code</a>&nbsp
        <a href="https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/mixquant">🧪 Examples</a>
    </div>


.. contents:: Table of Contents
   :local:
   :depth: 3


About the Algorithm
-------------------

In few-bit PTQ, activation outliers inflate the dynamic range, often decreasing the resolution of the quantizer and 
increasing its resulting rounding error. Rotation-based PTQ methods reduce dynamic range by diffusing large values across 
vector coordinates before quantization.

Recent methods [1,2] use block rotations, which apply independent rotations to fixed-size partitions of an activation 
vector. For hidden dimension :math:`d = nb` with :math:`n` blocks of size :math:`b`, block rotations reduce the compute 
requirements of Hadamard matrices from :math:`O(d \log d)` to :math:`O(d \log b)`, which can materially reduce 
inference overhead.

However, the outlier suppression behavior of Hadamard rotations changes under block structure, as seen below.

.. figure:: https://github.com/user-attachments/assets/9f01f26d-9f96-4fc2-a6e8-b52c7e7f4fca
   :alt: Input activation distributions vs block rotation size.
   :align: center
   :width: 100%

   Input activation distributions sampled from 2048 tokens of WikiText2 at the third down projection layer in Llama3 1B 
   under four configurations: (a) original model, (b) block Hadamard rotation with :math:`b = 32`, (c) block Hadamard 
   rotation with :math:`b = 128`, and (d) full-vector rotation. As :math:`b \to d`, the activation range decreases, 
   showing smaller blocks can be less effective at suppressing outliers.

Why block rotations degrade at small block sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For activation vector :math:`X \in \mathbb{R}^d` and block rotation :math:`\tilde{R} = diag(R, \ldots, R)` with 
Hadamard matrix :math:`R \in \mathbb{R}^{b \times b}`, Proposition 3.3 in the paper shows that the maximum post-rotation 
magnitude under a block Hadamard rotation satisfies

.. math::

   \|X \tilde{R}\|_\infty
   \le \max_{j \in [n]} \delta_{\{j\}} \sqrt{b}\, \|X_{\{j\}}\|_\infty,

where :math:`\delta_{\{j\}}` is the activation mass concentration of block :math:`j`, defined as

.. math::

   \delta_{\{j\}} =
   \frac{\|X_{\{j\}}\|_1}{b \|X_{\{j\}}\|_\infty}.

Since :math:`\|X_{\{j\}}\|_\infty \le \|X_{\{j\}}\|_1 \le b \|X_{\{j\}}\|_\infty`, we have :math:`\delta_{\{j\}} \in [1/b, 1]`. Values near :math:`1` indicate a block with near-uniform magnitudes, while values near :math:`1/b` indicate a block dominated by a small number of large coordinates (i.e., stronger outliers).

.. admonition:: Key idea 💡

   For fixed :math:`b`, the deterministic worst-case bound is governed by the block(s) with the largest blockwise 
   :math:`\ell_1` mass (equivalently, the largest :math:`\delta_{\{j\}}\sqrt{b}\|X_{\{j\}}\|_\infty` term). As :math:`b` 
   decreases, fewer coordinates contribute to each rotated output, so any block that carries a large fraction of the 
   activation mass will dominate the post-rotation range.

Worst-case vs typical behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The bound above is deterministic and worst-case, so it is intentionally pessimistic.

To reason about the typical (non-adversarial) behavior, Proposition 3.5 in the paper analyzes block Hadamard rotations
under a mild sign-randomness model (activation signs modeled as i.i.d. Rademacher variables). Under this model, with
probability at least :math:`1 - \varepsilon`,

.. math::

   \|X \tilde{R}\|_\infty
   \;\le\;
   \sqrt{\frac{2}{b} \log\!\left(\frac{2d}{\varepsilon}\right)} \; \|X\|_2.

This explains a common empirical trend: as block size :math:`b` increases, post-rotation outliers typically decrease (with 
diminishing returns), but the online rotation cost increases as :math:`O(d \log b)`. MixQuant targets the small block size 
regime by improving the pre-rotation geometry (i.e., balancing blockwise mass) so that block rotations behave more like
their full-vector counterpart at the same compute budget.

MixQuant: block rotation-aware permutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MixQuant addresses this limitation by inserting a permutation :math:`P` to explicitly minimize the maximum per-block
:math:`\ell_1` mass before block rotation. A simple algebraic rewrite makes the optimization target explicit:

.. math::

   \delta_{\{j\}} \sqrt{b} \|X_{\{j\}}\|_\infty
   =
   \sqrt{b}
   \cdot
   \frac{\|X_{\{j\}}\|_1}{b \|X_{\{j\}}\|_\infty}
   \cdot
   \|X_{\{j\}}\|_\infty
   =
   \frac{\|X_{\{j\}}\|_1}{\sqrt{b}}.

Therefore, for fixed block size :math:`b`, the deterministic bound is governed by :math:`\max_{j \in [n]} \|X_{\{j\}}\|_1`.

Using activation statistics, the permutation is calibrated so that large-magnitude coordinates are distributed across 
blocks rather than concentrated in a small subset of them. After permutation, the per-block :math:`\ell_1` norms are 
more balanced, which tightens the above bound and improves outlier suppression for a fixed block size.

In Brevitas, the default permutation strategy is MassDiff, the greedy calibration algorithm
described in Algorithm 1 of the paper. Intuitively, MassDiff greedily assigns channels to blocks so as to equalize (in 
expectation) the blockwise :math:`\ell_1`  mass. Given a calibration dataset, MassDiff:

1. Computes an average magnitude score for each channel
2. Processes channels in descending order of score
3. Assigns each channel to the block whose accumulated :math:`\ell_1` mass would increase the least
4. Continues until all blocks reach size :math:`b`

The resulting permutations are then merged into surrounding weights, as we'll explain next.

Permutation-equivariant regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: https://github.com/user-attachments/assets/fba5f64b-71b6-4f85-b443-2f1e134485bb
   :alt: Example quantization graph architecture for a standard transformer block.
   :align: center
   :width: 100%

   An illustration of a quantization graph architecture for a standard transformer block, merging rotations and permutations 
   wherever possible and quantizing the weights and activations for all linear layers

A permutation-equivariant region is any subgraph whose operations commute with feature-wise permutations.
Within such regions, permutation :math:`P` can be commuted through subgraph :math:`\Phi` and absorbed into surrounding 
weights :math:`W_1, W_2`. For example,

.. math::

   \Phi(X W_1) W_2
   =
   \Phi(X W_1 P)\, P^T W_2.

In transformer blocks, this typically includes compositions of:

- Linear → elementwise activation (SiLU / GELU / ReLU)
- Elementwise multiplication (e.g., SwiGLU gating)
- Residual addition (when both branches share the same permutation)
- Other featurewise operations that do not mix hidden coordinates

In contrast, operations that reshape or mix the hidden feature dimension generally break permutation equivariance and therefore delimit the region within which permutations can be merged.

Brevitas automatically detects these permutation-equivariant regions and merges calibrated permutations into adjacent linear 
weights prior to deployment. This ensures that no explicit permutation operator remains in the inference graph.

For the exact graph patterns and implementation details, see: ``brevitas.graph.permute`` `here <https://github.com/Xilinx/brevitas/blob/3719448d0da6e2b5815e9f977669dfd9fdebbdd8/src/brevitas/graph/permute.py>`_.

.. code:: pycon

   >>> import brevitas.graph.permute
   >>> brevitas.graph.permute._permute_invariant_layers
   (<class 'torch.nn.modules.activation.ReLU'>,
    <class 'torch.nn.modules.activation.LeakyReLU'>,
    <class 'torch.nn.modules.activation.GELU'>,
    <class 'torch.nn.modules.activation.SELU'>,
    <class 'torch.nn.modules.activation.SiLU'>,
    <class 'torch.nn.modules.normalization.RMSNorm'>)


Implementation Overview
--------------------

MixQuant is available through the LLM entry point ``brevitas_ptq_llm``.

At a high level, the Brevitas implementation of MixQuant:

1. Finds regions for rotations
2. Collects activation statistics to calibrate the permutation
3. Calibrates and merges permutations into surrounding weights
4. Inserts and merges rotations, leaving only the online rotations in the compute graph
5. Runs the rest of the PTQ pipeline (e.g., error correction, etc.)

The ``rotate_permute_mode`` context manager encapsulates most of the MixQuant workflow, as illustrated with the following 
pseudocode:

.. code:: python

   from brevitas.graph.equalize import GraphRotationEqualization
   from brevitas.graph.permute import rotate_permute_mode

   block_size = 32

   # class for finding regions for rotations
   rotation = GraphRotationEqualization(
       orphan_sink=True,  # enables online rotations
       rotation_block_size=block_size)

   with rotate_permute_mode(
       model,
       rotation=rotation,
       permute_fn='massdiff',
       block_size=block_size) as rpm:
      # 1. identifies regions for rotations on entry to context manager
      model = rpm.model
      with torch.no_grad():
         for data in dataloader:
            model(**data)  # 2. collects activation stats
      rewriters = rpm.rewriters
      # 3. calibrates and merges permutations on exit from context manager
   
   # 4. inserts and merges rotations
   model = apply_rewriters(model, rewriters)

   # 5.continue with the rest of the PTQ pipeline
   model = apply_qronos(model, ...)

See how this is used in 
`our LLM entry point <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/llm/main.py>`_!

The key CLI arguments are:

- ``--rotation``:
  Enables rotation equalization. If not specified, then neither rotations nor permutations are applied.
  
  Options:

  - ``fused_no_fx``: compute rewriters on a traced graph, then apply them to the model.
  - ``layerwise``: apply layerwise rotations; permutations are still merged.

- ``--rotation-block-size``:
  Block size :math:`b` for block Hadamard rotations (e.g. 16/32/64/128). Smaller = faster online rotation.
  If not set, the default behavior is to use full-vector rotations and permutations are not applied.

- ``--permute-fn``:
  Enables permutations if ``rotation`` and ``rotation_block_size`` are specified.
  
  Options:

  - ``massdiff`` (recommended)
  - ``zigzag``
  - ``absmax``
  - ``random``

  If unset or ``null``, no permutation is applied.

- ``--disable-block-rotation-for-fused``:
  If ``rotation`` and ``rotation_block_size`` are specified, keep fused rotations as full-vector and only use block 
  rotations where rotations remain online. This is a no-op if ``rotation_block_size`` is unset.


Getting Started
---------------

Install Brevitas and the required dependencies:

.. code::shell

   pip install brevitas[llm, export] lighteval


Below are the versions used for these results; different versions may yield different results.

- ``python==3.12``
- ``torch==2.6.0+rocm6.1``
- ``transformers==4.57.3``
- ``lighteval==0.13.0``

You can install PyTorch for ROCm 6.1 via:

.. code:: shell

   pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1


Quickstart
~~~~~~~~~~~

We provide pre-packaged config files in `brevitas_examples/papers/mixquant 
<https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/mixquant>`_ to enable similar experiments 
described in the paper. The provided configurations specify Llama-3.2-1B-Instruct, but you can specify different 
Huggingface models in the CLI args. For example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-mixquant_star-int4.yml --model=meta-llama/Llama-3.2-3B-Instruct
   
The MixQuant* config specifies:

- INT4 weights and activations
- Full-vector Hadamard rotations where mergable (i.e., R1 and R2)
- Online block Hadamard rotations with block size 32 (i.e., R3)
- MixQuant permutations via ``massdiff`` (i.e., P3)
- `Qronos <https://xilinx.github.io/brevitas/dev/papers/qronos.html>`_ rounding [3]


+-------+-----------+-----------+-------+-------+--------+-------+-------+
| Model | float_ppl | quant_ppl | ARC-C | ARC-E | HellaS | PIQA  | WinoG |
+=======+===========+===========+=======+=======+========+=======+=======+
| 1B    | 11.7      | 17.0      | 27.5  | 52.5  | 38.9   | 67.1  | 52.5  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+
| 3B    | 9.9       | 13.3      | 32.9  | 61.7  | 45.2   | 71.8  | 54.5  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+
| 8B    | 6.5       | 8.5       | 43.2  | 74.0  | 51.4   | 74.9  | 59.0  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+

You can override other hyperparamters via CLI. For example:

.. code:: shell

   # Try a different block size
   brevitas_ptq_llm --config=llama3-mixquant_star-int4.yml --rotation-block_size=16

   # Try a different permutation strategy
   brevitas_ptq_llm --config=llama3-mixquant_star-int4.yml --permute-fn=zigzag


The MixQuant† config specifies:

- INT4 weights and activations
- Learnable mergable rotations via CayleySGD (i.e., R1 and R2)
- Online block Hadamard rotations with block size 32 (i.e., R3)
- MixQuant permutations via ``massdiff`` (i.e., P3)
- Round-to-nearest (RTN) rounding

+-------+-----------+-----------+-------+-------+--------+-------+-------+
| Model | float_ppl | quant_ppl | ARC-C | ARC-E | HellaS | PIQA  | WinoG |
+=======+===========+===========+=======+=======+========+=======+=======+
| 1B    | 11.7      | 15.8      | 25.9  | 47.5  | 39.3   | 65.3  | 51.6  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+
| 3B    | 9.9       | 10.9      | 33.6  | 62.0  | 46.8   | 68.7  | 53.7  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+
| 8B    | 6.5       | 8.38      | 43.2  | 71.3  | 53.4   | 74.1  | 58.8  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+


To run multiple experiments in parallel across GPUs (e.g., sweeping block sizes), use the benchmark script:

.. code:: shell

   python benchmark.py --config benchmark-rotation_block_size.yml --results results/ --gpus 0,1

where ``--gpus`` refers to how many gpus to use. If multiple GPUs are specified, each one will be used to run an 
individual experiment. Below, we summarize results when quantizing Llama-3.2-1B-Instruct weights and activations to 
INT4 with and without MixQuant using MassDiff as the permutation algorithm.

+----------------------+------+------+------+------+------+------+------+
| Block Size           | 16   | 32   | 64   | 128  | 256  | 512  | Full |
+======================+======+======+======+======+======+======+======+
| No Permute           | 35.9 | 26.5 | 22.9 | 20.4 | 19.1 | 17.3 | 16.2 |
+----------------------+------+------+------+------+------+------+------+
| MixQuant             | 18.2 | 17.0 | 16.6 | 16.1 | 16.1 | 15.9 | 16.2 |
+----------------------+------+------+------+------+------+------+------+


Citation
--------

.. code:: bibtex

   @article{sanjeet2026mixquant,
     title   = {MixQuant: Pushing the Limits of Block Rotations in Post-Training Quantization},
     author  = {Sai Sanjeet and Ian Colbert and Pablo Monteagudo-Lago and Giuseppe Franco and Yaman Umuroglu and Nicholas J. Fraser},
     year    = {2026},
     eprint  = {2601.22347},
     archivePrefix = {arXiv},
     primaryClass  = {cs.LG},
     url     = {https://arxiv.org/abs/2601.22347},
   }

Note that this tutorial is not intended to reproduce all the experiments from the original 
paper. To more accurately reproduce experiments from the paper, please see `this 
<https://github.com/i-colbert/brevitas/tree/mixquant/src/brevitas_examples/papers/mixquant>`_ branch.

References
----------

[1] Egiazarian, V., et al. *Bridging the gap between promise and performance for microscaling FP4 quantization.* ICLR (2026).

[2] Shao, Y., et al. *Block rotation is all you need for MXFP4 quantization.* arXiv preprint (2025).

[3] Zhang, S., et al. *Qronos: Correcting the past by shaping the future... in post-training quantization.* ICLR (2026).
