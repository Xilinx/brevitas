Pushing the Limits of Block Rotations in Post-Training Quantization
===================================================================

PeRQ is a post-training quantization (PTQ) technique that improves the outlier suppression
capabilities of block Hadamard rotations. Prior to rotation, PeRQ inserts calibrated
permutations to redistribute activation mass within permutation-equivariant regions of the graph.
The permutations are calibrated once offline using activation statistics and then merged into
surrounding weight tensors before deployment so they do not incur additional inference overhead.
See the `paper <https://arxiv.org/abs/2601.22347>`_ for the full theoretical treatment!

.. raw:: html

    <div align="center">
        <a href="https://arxiv.org/abs/2601.22347">📄 Paper</a>&nbsp
        <a href="https://github.com/Xilinx/brevitas/blob/dev/src/brevitas/graph/permute.py">💻 Code</a>&nbsp
        <a href="https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/perq">🧪 Examples</a>
    </div>


.. contents:: Table of Contents
   :local:
   :depth: 3


About the Algorithm
-------------------

In few-bit PTQ, activation outliers inflate the dynamic range, often decreasing the resolution of
the quantizer and increasing its resulting rounding error. Rotation-based PTQ methods reduce
dynamic range by diffusing large values across vector coordinates before quantization.

Recent methods [1,2] use block rotations, which apply independent Hadamard transformations to
fixed-size partitions of an activation vector. For hidden dimension :math:`d = nb` with :math:`n`
blocks of size :math:`b`, block rotations reduce the compute requirements from
:math:`O(d \log d)` to :math:`O(d \log b)`, which can materially reduce inference overhead.

However, the outlier suppression behavior changes under block structure, as seen below.

.. figure:: https://github.com/user-attachments/assets/9f01f26d-9f96-4fc2-a6e8-b52c7e7f4fca
   :alt: Input activation distributions vs block rotation size.
   :align: center
   :width: 100%

   Input activation distributions sampled from 2048 tokens of WikiText2 at the third down
   projection layer in Llama3 1B under four configurations: (a) original model, (b) block
   Hadamard rotation with :math:`b = 32`, (c) block Hadamard rotation with :math:`b = 128`,
   and (d) full-vector rotation. As :math:`b \to d`, the activation range decreases, showing
   smaller blocks can be less effective at suppressing outliers.

The core issue is that when outlier channels are concentrated within the same block, a block
rotation cannot diffuse them effectively — they stay within that block rather than spreading
across the full hidden dimension. PeRQ shows that
worst-case post-rotation outliers are governed by the maximum per-block :math:`\ell_1` mass,
which tightens as block mass becomes more balanced.

PeRQ addresses this by inserting a permutation :math:`P` before rotation to explicitly
balance the per-block :math:`\ell_1` mass across blocks. The default calibration algorithm,
MassDiff (Algorithm 1 in the paper), is a greedy assignment that:

1. Computes an average magnitude score for each channel
2. Processes channels in descending order of score
3. Assigns each channel to the block whose accumulated :math:`\ell_1` mass would increase the least
4. Continues until all blocks reach size :math:`b`

The result is a permutation that spreads high-magnitude channels evenly across blocks,
improving outlier suppression for a fixed block size and at no inference cost. The permutation
itself is never applied at runtime — it is absorbed into adjacent weight tensors during
calibration, as described in the next section.

Permutation-equivariant regions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: https://github.com/user-attachments/assets/fba5f64b-71b6-4f85-b443-2f1e134485bb
   :alt: Example quantization graph architecture for a standard transformer block.
   :align: center
   :width: 100%

   An illustration of a quantization graph architecture for a standard transformer block,
   merging rotations and permutations wherever possible and quantizing the weights and
   activations for all linear layers.

The key to zero inference overhead is that permutations can be fused into weights wherever
the surrounding graph is permutation-equivariant, i.e., wherever the subgraph :math:`\Phi`
commutes with feature-wise permutations:

.. math::

   \Phi(X W_1) W_2 = \Phi(X W_1 P)\, P^T W_2.

In transformer blocks, this includes compositions of linear layers with elementwise activations
(SiLU, GELU), elementwise multiplications (e.g., SwiGLU gating), and residual additions where
both branches share the same permutation. Brevitas automatically detects these regions and
absorbs the permutation into adjacent weights, so no explicit permutation operator remains in
the inference graph.


Implementation Overview
-----------------------

The ``rotate_permute_mode`` context manager in
`brevitas.graph.permute <https://github.com/Xilinx/brevitas/blob/dev/src/brevitas/graph/permute.py>`_
encapsulates the full PeRQ workflow:

.. code:: python

   from brevitas.graph.equalize import GraphRotationEqualization, apply_rewriters
   from brevitas.graph.permute import rotate_permute_mode

   block_size = 32

   # Finds regions for rotations; delay_rewriters and return_rewriters
   # are required by rotate_permute_mode
   rotation = GraphRotationEqualization(
       orphan_sink=True,       # enables online rotations for residual sinks
       rotation_block_size=block_size,
       delay_rewriters=True,
       return_rewriters=True)

   with rotate_permute_mode(
           model,
           rotation=rotation,
           permute_fn='massdiff',
           block_size=block_size) as rpm:
       # on __enter__: rotation regions are identified; permutation hooks are set up
       model = rpm.model
       with torch.no_grad():
           for data in dataloader:
               model(**data)  # collects per-region activation statistics
       rewriters = rpm.rewriters
       # on __exit__: permutations are calibrated and fused into weights

   # apply the delayed rotation rewriters to the original model
   model = apply_rewriters(model, rewriters)

   # continue with the rest of the PTQ pipeline
   model = apply_qronos(model, ...)

On entry, ``rotate_permute_mode`` calls ``GraphRotationEqualization.apply()`` on a traced FX
graph to identify rotation regions and build the delayed rotation rewriters. It then registers
forward hooks on the sink modules of each permutation-equivariant region to collect activation
statistics during the calibration pass. On exit, it runs the calibration algorithm to produce
one permutation per region and fuses each permutation directly into the surrounding weights.

``GraphPermutationEqualization`` handles the graph walk and permutation application. It reuses
the same region-walk logic as used for rotation equalization to find permutation-equivariant regions.

The PeRQ pipeline is accessible via the ``brevitas_ptq_llm`` entrypoint. Key flags:

- ``--rotation-block-size`` — block size :math:`b` for online Hadamard rotations (e.g. ``32``).
  Omit for full-vector rotations (permutations are not applied in that case).
- ``--permute-fn`` — permutation strategy (``massdiff``, ``zigzag``, ``absmax``, ``random``).
  Omit or set to ``null`` to disable permutations.
- ``--disable-block-rotation-for-fused`` — apply block rotations only to online (orphan-sink)
  rotations; keep fused rotations as full-vector. This is the setting used by PeRQ\*.

A minimal example:

.. code:: bash

   brevitas_ptq_llm --model meta-llama/Llama-3.2-1B \
       --rotation-block-size 32 \
       --permute-fn massdiff \
       --disable-block-rotation-for-fused

See the `README <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/perq>`_
for full configuration details and benchmarking instructions.

Permutation strategies
~~~~~~~~~~~~~~~~~~~~~~

Four strategies are currently available, all registered via ``@register_permutation_method``:

- ``massdiff`` *(default, recommended)* — greedy assignment minimizing maximum per-block
  :math:`\ell_1` mass. The core loop:

  .. code:: python

     scores = torch.abs(x).mean(dim=0)        # average magnitude per channel
     _, indexes = torch.sort(scores, descending=True)
     block_norm = ...                          # initialize with top-n channels
     for i in indexes[num_blocks:]:
         # assign channel i to the block that minimizes the increase in l1 norm
         norms_after = block_norm + torch.abs(x[:, i]).unsqueeze(1)
         min_block = torch.argmin(norms_after.mean(dim=0))
         block_norm[:, min_block] += torch.abs(x[:, i])

- ``zigzag`` — sorts channels by magnitude then interleaves them in a zigzag pattern
  across blocks, alternating direction each row to balance adjacent-block norms.
- ``absmax`` — sorts channels by absolute maximum and assigns them in descending order
  (no block-awareness; simpler baseline).
- ``random`` — random shuffling; useful as a lower-bound baseline.

Furthermore, custom strategies can be registered:

.. code:: python

   from brevitas.graph.permute import register_permutation_method

   @register_permutation_method("my_permute")
   def my_permute(x, block_size):
       # x: [tokens, channels], block_size: int
       # return: index tensor of shape [channels]
       return torch.arange(x.shape[-1])


Results
-------

The results below use Llama-3.2 Instruct models with INT4 weight-activation quantization.
See the `README <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/perq>`_
for instructions on how to reproduce. Below are the versions used; different versions 
may yield different results.

- ``python==3.12``
- ``torch==2.6.0+rocm6.1``
- ``transformers==4.57.3``
- ``lighteval==0.13.0``

PeRQ\* (block rotations + MassDiff + Qronos)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full-vector Hadamard rotations where mergeable (R1, R2), online block Hadamard rotations with
block size 32 (R3), MassDiff permutations (P3), and `Qronos <https://xilinx.github.io/brevitas/dev/papers/qronos.html>`_ rounding [3].

+-------+-----------+-----------+-------+-------+--------+-------+-------+
| Model | float_ppl | quant_ppl | ARC-C | ARC-E | HellaS | PIQA  | WinoG |
+=======+===========+===========+=======+=======+========+=======+=======+
| 1B    | 11.7      | 17.0      | 27.5  | 52.5  | 38.9   | 67.1  | 52.5  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+
| 3B    | 9.9       | 13.3      | 32.9  | 61.7  | 45.2   | 71.8  | 54.5  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+
| 8B    | 6.5       | 8.5       | 43.2  | 74.0  | 51.4   | 74.9  | 59.0  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+

PeRQ† (learned rotations + MassDiff + RTN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Learnable mergeable rotations via CayleySGD (R1, R2), online block Hadamard rotations with
block size 32 (R3), MassDiff permutations (P3), and round-to-nearest rounding.

+-------+-----------+-----------+-------+-------+--------+-------+-------+
| Model | float_ppl | quant_ppl | ARC-C | ARC-E | HellaS | PIQA  | WinoG |
+=======+===========+===========+=======+=======+========+=======+=======+
| 1B    | 11.7      | 15.8      | 25.9  | 47.5  | 39.3   | 65.3  | 51.6  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+
| 3B    | 9.9       | 10.9      | 33.6  | 62.0  | 46.8   | 68.7  | 53.7  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+
| 8B    | 6.5       | 8.38      | 43.2  | 71.3  | 53.4   | 74.1  | 58.8  |
+-------+-----------+-----------+-------+-------+--------+-------+-------+


Citation
--------

.. code:: bibtex

   @article{sanjeet2026perq,
     title   = {Pushing the Limits of Block Rotations in Post-Training Quantization},
     author  = {Sai Sanjeet and Ian Colbert and Pablo Monteagudo-Lago and Giuseppe Franco and Yaman Umuroglu and Nicholas J. Fraser},
     year    = {2026},
     eprint  = {2601.22347},
     archivePrefix = {arXiv},
     primaryClass  = {cs.LG},
     url     = {https://arxiv.org/abs/2601.22347},
   }

Note that this page is not intended to reproduce all experiments from the original paper.
To more accurately reproduce the paper's experiments, please see
`this branch <https://github.com/i-colbert/brevitas/tree/permutations/src/brevitas_examples/papers/perq>`_.

References
----------

[1] Egiazarian, V., et al. *Bridging the gap between promise and performance for microscaling FP4 quantization.* ICLR (2026).

[2] Shao, Y., et al. *Block rotation is all you need for MXFP4 quantization.* arXiv preprint (2025).

[3] Zhang, S., et al. *Qronos: Correcting the past by shaping the future... in post-training quantization.* ICLR (2026).
