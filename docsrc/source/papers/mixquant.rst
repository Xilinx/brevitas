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

In few-bit PTQ, activation outliers inflate the dynamic range, decreasing the resolution of the quantizer and its 
resulting rounding error. Rotation-based PTQ methods reduce dynamic range by diffusing large values across vector 
coordinates before quantization.

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
Hadamard matrix :math:`R \in \mathbb{R}^{b \times b}`, Proposition 3.3 in the paper shows that the post-rotation 
maximum magnitude under a block Hadamard rotation satisfies

.. math::

   \|X \tilde{R}\|_\infty
   \le \max_{j \in [n]} \delta_{\{j\}} \sqrt{b}\, \|X_{\{j\}}\|_\infty,

where :math:`\delta_{\{j\}}` is the activation mass concentration of block :math:`j`, defined as

.. math::

   \delta_{\{j\}} =
   \frac{\|X_{\{j\}}\|_1}{b \|X_{\{j\}}\|_\infty}.

.. admonition:: Key idea 💡

   For fixed :math:`b`, worst-case post-rotation outliers are governed by the block(s) with the largest mass. As 
   :math:`b` decreases, fewer coordinates contribute to each rotated value. If the mass of an activation vector is 
   concentrated in only a few blocks, then large-magnitude coordinates are not diffused as effectively.

MixQuant: block rotation-aware permutations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MixQuant addresses this limitation by inserting a permutation :math:`P` to explicitly minimize the maximum 
:math:`\ell_1` mass before block rotation.

Using activation statistics, the permutation is calibrated so that large-magnitude coordinates are distributed across 
blocks rather than concentrated in a small subset of them. After permutation, the per-block :math:`\ell_1` norms are 
more balanced, which tightens the bound above and improves outlier suppression for a fixed block size.

In Brevitas, the default permutation strategy is MassDiff, the greedy calibration algorithm
described in Algorithm 1 of the paper. Given a calibration dataset, MassDiff:

1. Computes an average magnitude score for each channel
2. Processes channels in descending order of score
3. Assigns each channel to the block whose accumulated :math:`\ell_1` mass would increase the least
4. Continues until all blocks reach size :math:`b`

Intuitively, this greedily balances the per-block mass over a calibration set, directly minimizing 
:math:`\delta_{\{j\}}` to improve outlier suppression.

Permutation-equivariant regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a pre-trained model, a permutation does not change model behavior if it is inserted inside a 
permutation-equivariant region (Definition 4.1 in the paper), i.e., a subgraph whose operations commute with 
permutations along the feature dimension.

Within transformer architectures, many subgraphs are permutation-equivariant along the feature dimension, 
including compositions of:

- linear → elementwise activation (e.g. SiLU/Swish) → elementwise multiply  
- residual additions  
- other featurewise operations  

Within such regions, permutation :math:`P` can be commuted through subgraph  :math:`\Phi` and absorbed into surrounding 
weights :math:`W_1, W_2`:

.. math::

   \Phi(X W_1) W_2
   =
   \Phi(X W_1 P)\, P^T W_2.

.. admonition:: Key idea 💡

   Brevitas identifies permutation-equivariant regions and merges the calibrated permutation into adjacent linear weights before deployment. As a result, no explicit permutation operator remains in the inference graph. Only rotations that are intentionally kept online incur runtime cost.

Implementation Overview
--------------------

MixQuant is available through the LLM entry point ``brevitas_ptq_llm``.

At a high level, the Brevitas implementation of MixQuant:

1. Finds regions for rotations
2. Collects activation statistics to calibrate the permutation
3. Calibrates and merges permutations into surrounding weights
4. Inserts and merges rotations, leaving only the online rotations in the compute graph
5. Runs the rest of the PTQ pipeline (e.g., error correction, etc.)

The ``rotate_permute_mode`` context manager encapsulates most of the MixQuant workflow:

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
            model(**data)  # 2. collecting activation stats
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
- ``fast_hadamard_transform==1.0.4`` (custom fork, see below)

You can install PyTorch for ROCm 6.1 via:

.. code:: shell

   pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1

You can install and build a fork of the ``fast_hadamard_transform`` library with ROCm support via:

.. code:: shell

   git clone https://github.com/jeffdaily/fast-hadamard-transform -b rocm
   cd fast-hadamard-transform
   pip install -e . --no-build-isolation


Quickstart
~~~~~~~~~~~

We provide pre-packaged config files in `brevitas_examples/papers/mixquant 
<https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/mixquant>`_ to enable similar experiments 
described in the paper. The provided configurations specify Llama-3.2-1B-Instruct, but you can specify different 
Huggingface models in the CLI args. For example:

.. code:: shell

   brevitas_ptq_llm --config=llama3-mixquant-int4.yml --model=meta-llama/Llama-3.2-3B-Instruct

The default config includes (among others):

- INT4 weights and activations
- Merge full-vector Hadamard rotations where possible
- Online block Hadamard rotations with block size 32
- MixQuant permutations via ``massdiff``
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
   brevitas_ptq_llm --config=llama3-mixquant-int4.yml --rotation-block_size=16

   # Try a different permutation strategy
   brevitas_ptq_llm --config=llama3-mixquant-int4.yml --permute-fn=zigzag


To run multiple experiments in parallel across GPUs (e.g., sweeping block sizes), use the benchmark script:

.. code:: shell

   python benchmark.py --config benchmark-rotation_block_size.yml --results results/ --gpus 0,1

where ``--gpus`` refers to how many gpus to use. If multiple GPUs are specified, each one will be used to run an 
individual experiment. Below, we summarize results when quantizing Llama-3.2-1B-Instruct weights and activations to 
INT4 with and without MixQuant using MassDiff as the permutation algorithm.

+----------------------+------+------+------+------+------+------+
| Block Size           | 16   | 32   | 64   | 128  | 256  | 512  |
+======================+======+======+======+======+======+======+
| No Permute           | 37.6 | 26.5 | 22.7 | 21.0 | 18.7 | 17.5 |
+----------------------+------+------+------+------+------+------+
| MixQuant             | 18.0 | 17.2 | 16.9 | 15.9 | 16.1 | 16.0 |
+----------------------+------+------+------+------+------+------+


Citation
--------

::

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

[2] Shao, Y., et al. *Block rotation is all you need for mxfp4 quantization.* arXiv preprint (2025).

[3] Zhang, S., et al. *Qronos: Correcting the past by shaping the future... in post-training quantization.* ICLR (2026).
