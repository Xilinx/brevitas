====================
Learned Round
====================

Learned Round is a **post-training quantization (PTQ)** technique that improves quantization quality by **learning per-weight
rounding decisions**, instead of relying on fixed round-to-nearest (RTN). It unifies methods such as **AdaRound** [1]_ and
**SignRound** [2]_ under a single, configurable framework integrated into Brevitas' PTQ pipelines.

.. raw:: html

    <div align="center">
		<a href="https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/learned_round">💻 Examples</a>
    </div>

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
                                    "total_iters": 200}))],
                    batch_size=8,
                    iters=200,
                    losses_args=[LossArgs(cls="mse")],
                    loss_scaling_factor=1000.0,
                    use_best_model=True,
                    use_amp=True,
                    amp_dtype="float16",
                    fast_update=False),
                training_handlers=[
                    HandlerSpec(
                        name="learned_round",
                        config=LearnedRoundArgs(
                            learned_round_param=LearnedRoundImplType.IDENTITY))])))


Entrypoint Integration
~~~~~~~~~~~~~~~~~~~~~~

Learned Round is built into Brevitas' LLM and ImageNet PTQ entrypoints. When using these entrypoints,
caches, block forward functions, and block extraction logic are handled internally so you only need to
pass the appropriate CLI flags.

**LLM entrypoint.** The ``brevitas_ptq_llm`` command enables learned round through the
``--learned-round`` flag (currently accepts ``identity``). The ``--gpxq-block-name`` flag must be set
to the transformer block attribute path (e.g., ``model.model.layers`` for Qwen and LLaMA‑family
models). The following example applies SignRound‑style learned round to a Qwen model:

.. code-block:: bash

    brevitas_ptq_llm \
        --model Qwen/Qwen3-1.7B \
        --learned-round identity \
        --gpxq-block-name model.model.layers \
        --learned-round-iters 200 \
        --learned-round-lr 5e-3 \
        --weight-bit-width 4 \
        --weight-quant-granularity per_group \
        --weight-group-size 128

**ImageNet entrypoint.** The ``brevitas_ptq_imagenet_val`` command supports learned round through the
``--learned-round`` flag, which accepts ``identity``, ``sigmoid``, or ``hard_sigmoid``. The
``--target-backend layerwise`` flag is required. The loss function can be set with
``--learned-round-loss`` (choices: ``regularised_mse``, ``mse``; default: ``regularised_mse``).
The following example uses an AdaRound‑style configuration with sigmoid rounding and regularized MSE:

.. code-block:: bash

    brevitas_ptq_imagenet_val \
        --calibration-dir /path/to/imagenet/train \
        --validation-dir /path/to/imagenet/val \
        --learned-round sigmoid \
        --target-backend layerwise \
        --learned-round-mode layerwise \
        --learned-round-loss regularised_mse \
        --learned-round-iters 1000 \
        --learned-round-lr 1e-3

When using ``regularised_mse``, the loss combines MSE with AdaRound's [1]_ round regularization term
(defaults: weight ``0.01``, temperature annealing from ``20`` to ``2``, ``20%`` warmup).

More examples on how to use learned round through the LLM entrypoint are provided in
`LLM Learned Round Examples <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/learned_round>`_.


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

To use Learned Round with custom models or datasets outside of the supported entrypoints,
four components need to be defined. The following walks through each one and illustrates
them with a self-contained MLP example on synthetic data.


1. Model and blocks
^^^^^^^^^^^^^^^^^^^^

Learned Round optimizes the model one **block** at a time. A block is the **unit of
optimization**: a repeated structural pattern in the network architecture. Typical
examples include a ResNet block (conv‑bn‑relu‑conv‑bn) in vision models, an
Attention+MLP layer in transformer‑based LLMs, or even an individual layer when
fine‑grained per‑layer optimization is preferred. Blocks must be accessible as named
submodules so they can be extracted programmatically.

The model below is a 3‑block quantized MLP for regression. Each block contains two
``QuantLinear`` layers with a ``QuantReLU`` activation, named ``block_0`` through
``block_2``.

.. code-block:: python

    import torch
    from torch import nn
    import brevitas.nn as qnn

    class QuantBlock(nn.Module):
        def __init__(self, in_features, hidden_dim, out_features):
            super().__init__()
            self.linear1 = qnn.QuantLinear(in_features, hidden_dim, weight_bit_width=3)
            self.relu = qnn.QuantReLU(return_quant_tensor=True)
            self.linear2 = qnn.QuantLinear(hidden_dim, out_features, weight_bit_width=3)

        def forward(self, x):
            return self.linear2(self.relu(self.linear1(x)))

    class QuantMLP(nn.Module):
        def __init__(self, in_features, hidden_dim, out_features):
            super().__init__()
            self.block_0 = QuantBlock(in_features, hidden_dim, hidden_dim)
            self.block_1 = QuantBlock(hidden_dim, hidden_dim, hidden_dim)
            self.block_2 = QuantBlock(hidden_dim, hidden_dim, out_features)

        def forward(self, x):
            return self.block_2(self.block_1(self.block_0(x)))


2. Cache
^^^^^^^^

A **cache** captures block inputs and reference outputs during a calibration forward pass
so they can be replayed during optimization. It must inherit from ``Cache`` (a ``Dataset``
subclass defined in ``learned_round_utils.py``) and implement ``store_inputs``,
``store_output``, ``reset_cache``, ``__getitem__``, ``__len__``, and ``collate_fn``.
Inputs are typically split along the batch dimension so each sample is stored individually.

.. code-block:: python

    from typing import Any, Dict, Iterable, List, Tuple
    from brevitas_examples.common.learned_round.learned_round_utils import Cache

    class CacheMLP(Cache[torch.Tensor, torch.Tensor]):

        def __init__(self):
            self.inputs: List[torch.Tensor] = []
            self.outputs: List[torch.Tensor] = []

        def store_inputs(self, args: Tuple[torch.Tensor, ...], kwargs: Dict[str, Any]) -> None:
            self.inputs.extend(torch.split(args[0], 1, dim=0))

        def store_output(self, output: Any) -> None:
            self.outputs.extend(torch.split(output, 1, dim=0))

        def reset_cache(self) -> None:
            self.inputs, self.outputs = [], []

        def __len__(self) -> int:
            return len(self.inputs)

        def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
            return self.inputs[index], self.outputs[index]

        def collate_fn(
                self,
                batch: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            inputs, outputs = zip(*batch)
            return torch.cat(inputs, dim=0), torch.cat(outputs, dim=0)


3. Forward functions
^^^^^^^^^^^^^^^^^^^^

Two forward functions are needed:

- A **model forward function** (``ModelForwardFn`` protocol) that runs the full model on
  a calibration batch. This is used to populate the cache. Note that it receives raw
  batches from the ``DataLoader`` — for example, ``TensorDataset`` yields a list of
  tensors, so the input must be unpacked.
- A **block forward function** (``BlockForwardFn`` protocol) that runs a single block
  on cached inputs and returns its output, used during per‑block optimization.

.. code-block:: python

    from accelerate.utils.operations import send_to_device

    def mlp_forward(model: nn.Module, inputs: List[torch.Tensor]) -> None:
        device = next(model.parameters()).device
        # TensorDataset yields [tensor], so unpack the first element
        model(send_to_device(inputs[0], device))

    def mlp_block_forward(block: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
        device = next(block.parameters()).device
        return block(send_to_device(inputs, device))


4. Block extraction function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **block extraction function** returns the ordered list of blocks (units of optimization)
from the model. It typically delegates to ``get_blocks`` with a check function that
identifies blocks by name or type. The blocks are optimized sequentially in the order
returned.

.. code-block:: python

    from brevitas_examples.common.learned_round.learned_round_trainer import get_blocks

    def get_mlp_blocks(model: nn.Module) -> List[nn.Module]:
        return get_blocks(model, lambda module, name: name.startswith("block_"))


Putting it all together
^^^^^^^^^^^^^^^^^^^^^^^

With the four components defined, create a calibration ``DataLoader``, configure the
trainer with ``TrainerConfig``, and call ``trainer.train()`` to run block‑wise
optimization. The configuration below uses SignSGD with a linear LR decay, MSE loss, and
the identity (SignRound‑style) rounding parameterization.

.. code-block:: python

    from torch.utils.data import DataLoader, TensorDataset
    from brevitas_examples.common.learned_round.learned_round_trainer import LearnedRoundTrainer
    from brevitas_examples.common.learned_round.learned_round_args import (
        HandlerSpec, LossArgs, LRSchedulerArgs, OptimizerArgs,
        TrainerConfig, TrainingArgs)
    from brevitas_examples.common.learned_round.learned_round_method import LearnedRoundArgs

    # Synthetic calibration data
    calib_loader = DataLoader(TensorDataset(torch.randn(64, 8)), batch_size=8)

    # Model
    model = QuantMLP(in_features=8, hidden_dim=16, out_features=1)

    # Trainer configuration
    config = TrainerConfig(
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
                            "total_iters": 100}))],
            batch_size=8,
            iters=100,
            losses_args=[LossArgs(cls="mse")],
            loss_scaling_factor=1000.0),
        training_handlers=[
            HandlerSpec(
                name="learned_round",
                config=LearnedRoundArgs(learned_round_param="identity"))])

    # Run learned round optimization
    trainer = LearnedRoundTrainer(config=config)
    trainer.train(
        model=model,
        model_forward=mlp_forward,
        block_forward=mlp_block_forward,
        data_loader=calib_loader,
        cache=CacheMLP(),
        get_blocks_fn=get_mlp_blocks,
        keep_gpu=True)

Next Steps
----------

Learned Round has been evaluated in the LLM entrypoint across multiple quantization scenarios, including weight-only and weight-and-activation PTQ,
and in combination with outlier suppression techniques. For detailed results, as well as instructions on how to reproduce them, see ``brevitas_examples/papers/learned_round/README.md``.

References
----------

.. rubric::

.. [1] Nagel, M., Amjad, R. A., Van Baalen, M., Louizos, C., & Blankevoort, T. (2020, November). Up or down? adaptive rounding for post-training quantization. In International conference on machine learning (pp. 7197-7206). PMLR.
.. [2] Cheng, W., Zhang, W., Shen, H., Cai, Y., He, X., Kaokao, L., & Liu, Y. (2024, November). Optimize weight rounding via signed gradient descent for the quantization of llms. In Findings of the Association for Computational Linguistics: EMNLP 2024 (pp. 11332-11350).
.. [3] Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323.
.. [4] Zhang, S., Zhang, H., Colbert, I., & Saab, R. (2025). Qronos: Correcting the Past by Shaping the Future... in Post-Training Quantization. arXiv preprint arXiv:2505.11695.
.. [5] Ashkboos, S., Mohtashami, A., Croci, M. L., Li, B., Cameron, P., Jaggi, M., ... & Hensman, J. (2024). Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37, 100213-100240.
.. [6] Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Krishnamoorthi, R., ... & Blankevoort, T. (2024). Spinquant: Llm quantization with learned rotations. arXiv preprint arXiv:2405.16406.
.. [7] Zhang, A., Wang, N., Deng, Y., Li, X., Yang, Z., & Yin, P. (2024). Magr: Weight magnitude reduction for enhancing post-training quantization. Advances in neural information processing systems, 37, 85109-85130.
