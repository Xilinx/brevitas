====================
Learned Round
====================

Learned Round is a **post-training quantization (PTQ)** technique that improves quantization quality by **learning per-weight
rounding decisions**, instead of relying on fixed round-to-nearest (RTN). It unifies methods such as **AdaRound** [1]_ and
**SignRound** [2]_ under a single, configurable framework integrated into Brevitas’ PTQ pipelines.

.. contents:: Table of Contents
   :local:
   :depth: 3


About the Algorithm
-------------------

Motivation
~~~~~~~~~~

Quantization mappings generally **require a rounding operator**, for which **round‑to‑nearest (RTN)** is the standard choice.

For example, in symmetric integer quantization the mapping is typically written as:

.. math::

    \mathcal{Q}(W) := s \cdot \left(
    \text{clip}\left(
        \left\lceil \frac{W}{s} \right\rfloor + z,
        \min \mathcal{A}, \max \mathcal{A}
    \right) - z
    \right).

RTN is optimal when minimizing the **weight reconstruction error**

.. math::

    \lVert W - \mathcal{Q}(W) \rVert_2,

but this optimality does not generally hold when considering the **layer (or block‑wise) output reconstruction loss**

.. math::

    \lVert XW - X\mathcal{Q}(W) \rVert_2,

which is commonly used as a proxy for downstream accuracy degradation in PTQ.

This observation motivates **learned rounding**, where each weight is allowed to round **up or down** in a data‑driven way.

Rounding Optimization
~~~~~~~~~~~~~~~~~~~~~

Methods such as **AdaRound** [1]_ and **SignRound** [2]_ formulate rounding as a **binary optimization problem**, selecting
either the floor or the ceiling of the quantization grid for each weight. Although the resulting discrete problem is
NP‑hard, it can be relaxed into a **continuous optimization** by introducing learnable parameters inside the rounding
operator and optimizing them using calibration data.

In contrast to greedy solvers such as **GPTQ** [3]_ and **Qronos** [4]_, which typically solve closed‑form layer‑wise
objectives sequentially, learned rounding methods:

- jointly optimize rounding decisions (per layer or per block),
- rely on gradient‑based optimization over calibration data,
- restrict the search space to a limited subset of quantization grid points.

By jointly correcting quantization error across all weights within a block 
in a constrained manner, this approach more effectively reduces block output error 
while mitigating overfitting to calibration data. However, compared to **GPTQ** and **Qronos**, 
learned rounding typically requires greater compute and hyperparameter tuning.


Learned Round in Brevitas
-------------------------

In Brevitas, these approaches are unified under the name **Learned Round**, providing:

- a common abstraction for learned rounding,
- flexible choices of rounding parameterization and optimization strategy,
- seamless integration with existing PTQ pipelines (LLM and ImageNet entrypoints).

Learned Round is compatible with **all quantized data types currently supported by Brevitas**, including:

- integer quantization (e.g. INT2 / INT4 / INT8),
- weight‑only, weight‑and‑activation, and KV‑cache quantization,
- advanced formats such as **MXFP4**.

It is also composable with other PTQ techniques, including **QuaRot** [5]_, **SpinQuant** [6]_, and **MagR** [7]_.


Implementation Overview
~~~~~~~~~~~~~~~~~~~~~~~

At a high level, Learned Round performs **block‑wise post‑training optimization** of rounding decisions, 
following these steps:

1. Prepare the model (optional preprocessing, e.g. disabling internal caches).
2. Insert learnable rounding parameters into the quantization operators.
3. Decompose the model into blocks.
4. For each block:
   a. Cache block inputs (and reference outputs) using calibration data.  
   b. Optimize rounding parameters (and optionally scales) via a local reconstruction loss.  
   c. Freeze the optimized rounding decisions.
5. Optionally reuse cached activations to accelerate block‑to‑block optimization.
6. Restore the original model configuration for inference.

``LearnedRoundTrainer`` orchestrates this block‑wise optimization by wiring together:

- a learned rounding parameterization (e.g. ``LearnedRoundIdentity``),
- block‑level reconstruction losses (e.g. ``MSELoss``, ``RoundRegularisationLoss``),
- optimizers and learning‑rate schedulers,
- training configuration (batch size, iterations, AMP settings, etc.).


Following, an example configuration matching the **SignRound** [2]_ setup (without scale optimization) is provided:

.. code-block:: python
   :caption: `brevitas_examples/common/learned_round/learned_round_trainer.py`

    learned_round_trainer = LearnedRoundTrainer(
        config=Config(
            trainer=TrainerConfig(
                training_args=TrainingArgs(
                    optimizers_args=[
                        OptimizerArgs(
                            target_params="learned_round",
                            optimizer_cls="SignSGD",
                            lr=5e-3,
                            lr_scheduler_args=LRSchedulerArgs(
                                lr_scheduler_cls="LinearLR",
                                lr_scheduler_kwargs={
                                    "start_factor": 1.0,
                                    "end_factor": 0.0,
                                    "total_iters": 200,
                                },
                            ),
                        ),
                    ],
                    batch_size=8,
                    iters=200,
                    losses_args=[LossArgs(cls="mse")],
                    loss_scaling_factor=1000.0,
                    use_best_model=True,
                    use_amp=True,
                    amp_dtype="float16",
                    fast_update=False,
                ),
                training_handlers=[
                    HandlerSpec(
                        name="learned_round",
                        config=LearnedRoundArgs(
                            learned_round_param=LearnedRoundImplType.IDENTITY,
                        ),
                    )
                ],
            )
        )
    )


Entrypoint Integration
~~~~~~~~~~~~~~~~~~~~~~

Learned Round is available through Brevitas’ PTQ pipelines, including the LLM and ImageNet entrypoints. Therefore,
if you using Brevitas' entrypoints:

✅ You **do not need** to implement caches, block forward functions, or block extraction logic.

Lower‑level abstractions (cache objects, block forwards, etc.) are only required when building a **custom PTQ pipeline**
outside the supported entrypoints.

See:

- ``brevitas_examples/llm/llm_quant/learned_round_utils.py``
- ``brevitas_examples/imagenet_classification/ptq/learned_round_utils.py``


Extending Learned Round
-----------------------

Learned Round is designed to be extensible, supporting:

- custom learned‑round parameterizations,
- optimization of additional parameters (e.g. scales),
- integration with custom models and datasets.

This section targets advanced users.


Rounding Parameterizations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Learned Round expresses rounding as:

.. math::

    \text{round}(w) = \mathcal{R}(f(w; p)) + g(p), \quad
    \mathcal{R} \in \{\lfloor \cdot \rceil,\ \lfloor \cdot \rfloor,\ \lceil \cdot \rceil\},

where :math:`p` denotes learnable parameters controlling the rounding behavior
(typically only one of :math:`f` or :math:`g` is used).

Brevitas provides several implementations in
``brevitas/core/function_wrapper/learned_round.py``, including:

- **Sigmoid** (AdaRound‑style):

  .. math::

     \text{round}(p; w, T) = \lfloor w \rfloor + \sigma(p / T)

- **Identity** (SignRound‑style):

  .. math::

     \text{round}(p; w)
     = \left\lfloor w + \text{clip}(p, -0.5, 0.5) \right\rceil

To add a custom rounding parameterization:

1. Define a class implementing ``forward`` and ``round_forward`` similarly to existing implementations in
   ``brevitas/core/function_wrapper/learned_round.py``.
2. Register the implementation in:
   - ``LearnedRoundImplType`` (``brevitas/inject/enum.py``)
   - ``learned_round_impl`` (``brevitas/quant/solver/common.py``)


Extending to Custom Models or Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use Learned Round with custom models/datasets outside of the supported entrypoints, you will need to implement the following components:

1. **Cache**: A class, inheriting from ``brevitas_examples/common/learned_round/learned_round_utils.py:Cache``, that captures block inputs 
and reference outputs during the forward pass (see ``brevitas_examples/llm/llm_quant/learned_round_utils.py:CacheLLM`` for an example).

2. **Block forward function**: A function, implementing the `Protocol` ``brevitas_examples/common/learned_round/learned_round_utils.py:BlockForwardFn``, 
that performs the forward pass through the block being optimized, using the cached inputs and reference 
outputs (see ``brevitas_examples/llm/llm_quant/learned_round_utils.py:llm_block_forward`` for an example).

3. **Model forward function**: A function, implementing the `Protocol` ``brevitas_examples/common/learned_round/learned_round_utils.py:ModelForwardFn``, 
that performs a forward pass through the model (see ``brevitas_examples/llm/llm_quant/learned_round_utils.py:llm_forward`` for an example).

4. **Block extraction function**: A function that extracts the blocks to be optimized from the model 
(see ``brevitas_examples/llm/llm_quant/learned_round_utils.py:get_blocks`` for an example).

Getting Started
---------------

Learned Round has been evaluated in the LLM entrypoint across multiple quantization scenarios, including weight-only and weight-and-activation PTQ,
and in combination with outlier suppression techniques. For detailed results, as well as instructions on how to reproduce them, see ``brevitas_examples/papers/learned_round/README.md``.

.. rubric:: References

.. [1] Nagel, M., Amjad, R. A., Van Baalen, M., Louizos, C., & Blankevoort, T. (2020, November). Up or down? adaptive rounding for post-training quantization. In International conference on machine learning (pp. 7197-7206). PMLR. 
.. [2] Cheng, W., Zhang, W., Shen, H., Cai, Y., He, X., Kaokao, L., & Liu, Y. (2024, November). Optimize weight rounding via signed gradient descent for the quantization of llms. In Findings of the Association for Computational Linguistics: EMNLP 2024 (pp. 11332-11350).
.. [3] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323. 
.. [4] Zhang, S., Zhang, H., Colbert, I., & Saab, R. (2025). Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization. arXiv preprint arXiv:2505.11695. 
.. [5] Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P., Jaggi, M., ... & Hensman, J. (2024). Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37, 100213-100240.
.. [6] Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Krishnamoorthi, R., ... & Blankevoort, T. (2024). Spinquant: Llm quantization with learned rotations. arXiv preprint arXiv:2405.16406.
.. [7] Zhang, A., Wang, N., Deng, Y., Li, X., Yang, Z., & Yin, P. (2024). Magr: Weight magnitude reduction for enhancing post-training quantization. Advances in neural information processing systems, 37, 85109-85130.
.. [8] Shao, W., Chen, M., Zhang, Z., Xu, P., Zhao, L., Li, Z., ... & Luo, P. (2023). Omniquant: Omnidirectionally calibrated quantization for large language models. arXiv preprint arXiv:2308.13137.