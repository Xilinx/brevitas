Differentiable Mixed-Precision Assignment for Low-Precision Floating-Point Formats
==================================================================================

dMX is a differentiable mixed-precision quantization framework that *learns* the
per-layer floating-point bit-width of a network, focusing on the microscaling
floating-point (MXFP) family defined by the Open Compute Project (OCP). Instead
of applying a single MXFP format uniformly across every layer, dMX parameterizes
each layer's format with a single continuous, learnable offset that interpolates
between hardware-supported formats (e.g. MXFP4 and MXFP8) during calibration. A
temperature-based annealing schedule progressively discretizes these offsets so
the final assignment maps to a hardware-compatible format, while a target-aware
regularization term steers the average bit-width toward a user-specified budget.
Across Llama, Qwen3, and SmolLM2 models, dMX consistently yields
Pareto-dominating models and improves over KL divergence-based layer-selection
heuristics.

.. raw:: html

    <div align="center">
        <a href="https://arxiv.org/abs/2606.04115">📄 Paper</a>&nbsp
        <a href="https://github.com/Xilinx/brevitas/blob/dev/src/brevitas_examples/papers/dMX/learned_float_quantizer.py">💻 Code</a>&nbsp
        <a href="https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/dMX">🧪 Examples</a>
    </div>


.. contents:: Table of Contents
   :local:
   :depth: 3


How dMX Works
-------------

Picking a single low-precision format for the whole network is rarely the right
call. Push everything down to MXFP4 and quality tends to fall off a cliff; keep
everything at MXFP8 and you are paying for precision that most layers do not
need. The sweet spot lies in between — letting different layers use different
bit-widths — but finding that assignment is genuinely hard. The number of ways to
hand out formats grows exponentially with depth, and the layers interact:
quantization error introduced early on propagates and compounds through the rest
of the network in ways that are difficult to predict.

The usual shortcut is to score each layer in isolation — quantize it on its own,
measure how much some proxy like KL divergence degrades, and hand the most
sensitive layers more bits. It is cheap, but it is blind to exactly the
cross-layer interactions that make the problem hard in the first place.
Gradient-based search sidesteps that limitation by learning the bit-widths
end-to-end, and it has worked well for integer quantization. Floating point is
trickier, though. The format is not a single number: it splits into exponent and
mantissa bits (plus the exponent bias), and real hardware only supports a handful
of combinations. In the MX family, MXFP8, MXFP6, and MXFP4 map to ``E4M3``,
``E2M3``, and ``E2M1`` — so whatever a continuous search does during training, it
has to land on one of those discrete formats by the time you deploy.

dMX threads this needle by making the per-layer format a differentiable quantity,
learning it alongside the rest of the calibration, and gently forcing it back
onto a hardware-supported format as training winds down. Three ideas make that
possible.

The first is to stop treating the exponent and mantissa as independent knobs —
learning them separately could easily produce a format no accelerator can run.
Instead, dMX pins every supported format to a shared baseline and moves along it
with a single learnable offset :math:`\beta`:

.. math::

   E(2 + \beta)\,M(1 + \beta)

Here :math:`\beta = 0` recovers the baseline MXFP4 format (``E2M1``) and
:math:`\beta = 2` recovers MXFP8 (``E4M3``). During training :math:`\beta` is free
to roam anywhere in :math:`[0, 2]`, so a layer can drift smoothly between formats
rather than jumping between them. The quantizer math stays well-defined for
fractional bit-widths, which means gradients reach :math:`\beta` through ordinary
autograd — no special casing required. The same offset also drives an MXFP6/MXFP4
search (varying only the mantissa), and once the weight and activation bit-widths
are tied together, each layer is described by just one learnable scalar.

The second idea deals with the awkward fact that training likes continuous values
but inference demands discrete ones. The obvious fix — round to the nearest
format on every forward pass with a straight-through estimator — turns out to
backfire: layers flip back and forth between MXFP4 and MXFP8 from step to step,
and the training destabilizes. dMX avoids the thrashing by squashing the offset
through a temperature-controlled sigmoid instead:

.. math::

   \beta = F(\hat{\beta}, T) = \frac{2}{1 + e^{-T \cdot (\hat{\beta}/2 - 0.5)}}

where :math:`\hat{\beta}` is the raw learned offset and the temperature :math:`T`
climbs over the course of training. Early on, with :math:`T` small, the mapping
is nearly linear and the optimizer is free to explore the continuous space. As
:math:`T` rises, the curve sharpens into a step centered at :math:`\beta = 1`,
nudging offsets below 1 toward MXFP4 and offsets above 1 toward the
high-precision format. By the end, the format a layer *trains* with is the format
it will *run* with — no last-minute discretization shock.

The third idea gives the user a lever on the accuracy/efficiency trade-off. Left
to its own devices, the optimizer will happily spend bits, since more precision
almost always lowers the task loss. dMX keeps that in check with a target-aware
penalty that stays silent until the model's average bit-width drifts above a
budget you set:

.. math::

   \mathcal{R}_t = \max\!\left(0,\; \lambda \cdot (\bar{b}_{\text{current}} - \bar{b}_{\text{target}})\right)

The current average :math:`\bar{b}_{\text{current}}` can be a plain average over
layers or a tensor-size-weighted one that tracks actual memory cost, and
:math:`\bar{b}_{\text{target}}` is simply the budget you are aiming for. The nice
property is predictability: sweep :math:`\bar{b}_{\text{target}}` and you trace
out the whole accuracy/bit-width Pareto front point by point. If you would rather
not commit to a target, a simpler scaling penalty
:math:`\mathcal{R}_s = \lambda \cdot \bar{b}_{\text{current}}` is available too,
though it offers less direct control over where you land.

Because all of this amounts to adding one learnable offset per layer, dMX slots
into an existing PTQ pipeline rather than replacing it — in the experiments below
the bit-widths are learned *jointly* with SpinQuant-style Cayley-optimized
rotations.

🔍 See the `paper <https://arxiv.org/abs/2606.04115>`_ for the full continuous
float-conversion derivation and the closed-form bit-width gradients.


Results
-------

Experiments quantize all models with Brevitas, calibrating on 3200 samples from
FineWeb (batch size 8, 400 optimization steps) and evaluating WikiText-2
perplexity together with the average zero-shot accuracy over four reasoning
benchmarks (ARC-Challenge, ARC-Easy, HellaSwag, WinoGrande) via LightEval.
Rotations and bit-widths are optimized jointly, following a SpinQuant-like setup.
Below are the versions used; different versions may yield different results.

- ``python==3.12``
- ``torch==2.6.0``
- ``transformers==4.57.3``
- ``lighteval==0.13.0``

Each dMX row corresponds to a different target average bit-width; the baselines
apply a single MXFP format uniformly across all layers. The table below reports
the MXFP8/MXFP4 mixed-precision configuration.

+--------------+--------+-------------------+-----------+--------------+
| Model        | Type   | Avg Bit-Width     | Wiki2 (↓) | 0-shot (↑)   |
+==============+========+===================+===========+==============+
| Llama-3.2-1B | Float  | BF16              | 8.94      | 51.53        |
+              +--------+-------------------+-----------+--------------+
|              | MXFP4  | 4.0               | 11.68     | 47.82        |
+              +--------+-------------------+-----------+--------------+
|              | MXFP8  | 8.0               | 9.15      | 51.39        |
+              +--------+-------------------+-----------+--------------+
|              | dMX    | 4.57              | 11.02     | 48.11        |
+              +        +-------------------+-----------+--------------+
|              |        | 5.11              | 10.60     | 49.27        |
+              +        +-------------------+-----------+--------------+
|              |        | 6.04              | 9.83      | 49.61        |
+              +        +-------------------+-----------+--------------+
|              |        | 8.0               | 9.19      | 51.46        |
+--------------+--------+-------------------+-----------+--------------+
| SmolLM2-1.7B | Float  | BF16              | 7.61      | 59.52        |
+              +--------+-------------------+-----------+--------------+
|              | MXFP4  | 4.0               | 10.28     | 53.42        |
+              +--------+-------------------+-----------+--------------+
|              | MXFP8  | 8.0               | 7.86      | 58.69        |
+              +--------+-------------------+-----------+--------------+
|              | dMX    | 4.53              | 9.33      | 54.41        |
+              +        +-------------------+-----------+--------------+
|              |        | 5.09              | 8.92      | 55.36        |
+              +        +-------------------+-----------+--------------+
|              |        | 6.06              | 8.26      | 57.28        |
+              +        +-------------------+-----------+--------------+
|              |        | 7.29              | 7.96      | 57.90        |
+--------------+--------+-------------------+-----------+--------------+
| Qwen3-1.7B   | Float  | BF16              | 15.74     | 54.12        |
+              +--------+-------------------+-----------+--------------+
|              | MXFP4  | 4.0               | 12.57     | 51.15        |
+              +--------+-------------------+-----------+--------------+
|              | MXFP8  | 8.0               | 10.50     | 54.08        |
+              +--------+-------------------+-----------+--------------+
|              | dMX    | 4.53              | 11.91     | 52.20        |
+              +        +-------------------+-----------+--------------+
|              |        | 5.02              | 11.58     | 51.62        |
+              +        +-------------------+-----------+--------------+
|              |        | 5.90              | 11.02     | 53.74        |
+              +        +-------------------+-----------+--------------+
|              |        | 7.65              | 10.47     | 54.09        |
+--------------+--------+-------------------+-----------+--------------+

Even a small increase in average bit-width yields consistent gains in both
perplexity and zero-shot accuracy, and dMX closely matches the requested target
bit-width. The advantage is largest at *intermediate* bit-widths, where the
precision budget must be allocated carefully across layers — exactly the regime
in which end-to-end optimization outperforms KL divergence-based pre-selection.
Analyzing which layers are most often kept at high precision, the ``down_proj``
and ``v_proj`` projections emerge as the most sensitive to quantization across
models. The paper also reports an MXFP6/MXFP4 configuration — which turns out to
be more bit-width efficient still — and shows that these benefits carry over to
larger models, up to 8B parameters (see the paper for both).


How to Reproduce
----------------

The dMX pipeline is packaged as a self-contained example under
`brevitas_examples/papers/dMX <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/dMX>`_.
It is built entirely from two Brevitas *plugins* that register themselves into
Brevitas' registries when imported:

- ``learned_float_quantizer.py`` registers the ``learned_float`` quantizer
  (MXFP8/MXFP4, learning both exponent and mantissa via :math:`\beta`) and the
  ``mxfp6_learned_float`` quantizer (MXFP6/MXFP4, learning only the mantissa).
- ``custom_trainer.py`` registers the ``rotation_learned_bitwidth`` trainer,
  which jointly optimizes the Cayley-SGD rotation matrices and the SGD bit-width
  parameters, applies the temperature-annealing schedule, and adds the
  target-aware bit-width penalty.

The ``benchmark.py`` entrypoint imports both plugins (so they can be referenced
by bare name in the YAML) and sweeps the packaged configurations across models
and target bit-widths. For example:

.. code:: shell

   cd src/brevitas_examples/papers/dMX
   python benchmark.py --config benchmark/mxfp8_mixed_precision.yaml --results results/ --gpus 0,1

where ``--gpus`` is a comma-separated list of GPU device indices; each argument
combination runs on its own process/GPU. Use ``--dry-run`` to print the expanded
set of experiments without running them. The provided configurations are:

- ``benchmark/mxfp8_mixed_precision.yaml`` — MXFP8/MXFP4 sweep (``learned_float``).
- ``benchmark/mxfp6_mixed_precision.yaml`` — MXFP6/MXFP4 sweep (``mxfp6_learned_float``).
- ``benchmark/mxfp8_big_models_mixed_precision.yaml`` and
  ``benchmark/mxfp6_big_models_mixed_precision.yaml`` — the same sweeps tuned for
  larger (3B–8B) models.

The main knobs exposed by the ``rotation_learned_bitwidth`` trainer map directly
onto the paper's hyper-parameters:

- ``target_bit_width`` — the target average bit-width
  :math:`\bar{b}_{\text{target}}`; swept from 4.1 to 8 to trace the Pareto front.
- ``rotation_lr`` — learning rate for the Cayley-SGD rotation optimizer (1.5).
- ``bw_learning_rate`` — learning rate for the SGD bit-width optimizer (1 for the
  smaller models, 2 for the larger ones).
- ``delay_start`` — fraction of steps kept at constant (linear-like) temperature
  before annealing begins (:math:`T_{\text{ratio}} = 0.6`).
- ``simple_average_loss`` — whether :math:`\bar{b}_{\text{current}}` is the simple
  average or the tensor-size-weighted average across layers.

See the
`README <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/dMX>`_
for the full configuration reference.

Adapting dMX to a custom problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Nothing about dMX is hard-wired into Brevitas' core. The whole example is two
plugins — a quantizer and a trainer — that register themselves when imported and
are loaded through the LLM entrypoint's ``--custom-quantizer`` and
``--custom-trainer`` flags with the ``path/to/plugin.py:name`` syntax. Adapting
it to your own problem usually means writing a small ``.py`` file rather than
touching the library.

Say you want to target a different pair of formats. The ``learned_float``
quantizer is the template to copy: it drives the MXFP8/MXFP4 search by learning
both the exponent and the mantissa offset, and building a new quantizer is mostly
a matter of picking which parameters are learnable and over what range. To make
this concrete, the plugin already ships a variant, ``mxfp6_learned_float``, that
reuses the very same learned-bit-width machinery but freezes the exponent and
learns only the mantissa — the one change that turns an MXFP8/MXFP4 search into an
MXFP6/MXFP4 one.

The same recipe extends beyond floating point. Suppose you want dMX to choose,
per layer, between INT8 and INT4 integer formats. Integer quantizers have a
single ``bit_width`` to learn instead of separate exponent and mantissa fields,
so the plugin is even simpler: start from Brevitas' integer quantizers, mark the
bit-width as a learnable ``PARAMETER``, and let the shared ``RestrictBitWidth``
module anneal it between the two admissible values.

.. code:: python

   # my_int_quantizer.py — register with:
   #   --custom-quantizer my_int_quantizer.py:learned_int
   from brevitas.core.bit_width import BitWidthImplType
   from brevitas.quant.scaled_int import Int8ActPerTensorFloat
   from brevitas.quant.scaled_int import Int8WeightPerChannelFloat
   from brevitas.utils.python_utils import Registry
   from brevitas_examples.common.generative.quantizers import BaseQuantizer
   from brevitas_examples.common.generative.quantizers import QUANTIZERS_REGISTRY

   # Reuse the annealed, learnable bit-width restriction from the dMX plugin,
   # this time bounded between INT4 (low) and INT8 (high).
   from learned_float_quantizer import RestrictBitWidth


   class LearnedIntBitWidth:
       # Anneal the learnable bit-width between 4 and 8.
       min_bit_width = 4
       bit_width = 4
       bit_width_min_val = 4
       bit_width_max_val = 8
       restrict_bit_width_impl = RestrictBitWidth
       temperature = 0.4


   class Int4to8LearnedbitWeight(Int8WeightPerChannelFloat, LearnedIntBitWidth):
       bit_width_impl_type = BitWidthImplType.PARAMETER


   class Int4to8LearnedbitAct(Int8ActPerTensorFloat, LearnedIntBitWidth):
       bit_width_impl_type = BitWidthImplType.PARAMETER


   @Registry.register(QUANTIZERS_REGISTRY, "learned_int")
   class LearnedInt(BaseQuantizer):
       weight_quant = Int4to8LearnedbitWeight
       linear_input_quant = Int4to8LearnedbitAct

The ``rotation_learned_bitwidth`` trainer, the temperature schedule, and the
target-aware penalty all work unchanged — they only care that each layer exposes
a learnable bit-width offset, not whether the underlying format is floating point
or integer. The snippet above is illustrative rather than tuned; the exact ranges
and starting formats are the knobs you would adjust for your own hardware target.

Reshaping the objective is just as local. The bit-width penalty lives in
``custom_trainer.py``, so trading the target-aware penalty for the simple scaling
one — or for a cost that bakes in tighter memory or compute constraints — is a
small, self-contained edit, and the choice between the plain and
tensor-size-weighted average is already a flag (``simple_average_loss``). The
task loss is equally open: the trainer subclasses ``GeneralizedTrainer``, so you
can override the default cross-entropy loss with, say, a distillation loss and
register the result under a new name.

And there is no reason to stop at PTQ. Since the format offset is ultimately just
another learnable parameter, the same machinery drops naturally into other
regimes — parameter-efficient fine-tuning, or full QAT — with little more than a
different training loop around it.


Citation
--------

.. code:: bibtex

   @article{franco2026dmx,
         title={dMX: Differentiable Mixed-Precision Assignment for Low-Precision Floating-Point Formats},
         author={Franco, Giuseppe and Colbert, Ian and Monteagudo-Lago, Pablo and Marty, Felix and Fraser, Nicholas},
         journal={arXiv preprint arXiv:2606.04115},
         year={2026},
   }

Note that this page is not intended to reproduce every experiment from the
original paper. For the full set of experiments and configurations, see the
`examples directory <https://github.com/Xilinx/brevitas/tree/dev/src/brevitas_examples/papers/dMX>`_.

References
----------

[1] Darvish Rouhani, B., et al. *OCP microscaling formats (MX) specification.* Open Compute Project, 2023.

[2] Liu, Z., et al. *SpinQuant: LLM quantization with learned rotations.* 13th International Conference on Learning Representations, 2025.

[3] Yang, L., and Jin, Q. *FracBits: Mixed precision quantization via fractional bit-widths.* AAAI Conference on Artificial Intelligence, 2021.

[4] Huang, X., et al. *SDQ: Stochastic differentiable quantization with mixed precision.* 39th International Conference on Machine Learning, 2022.

[5] Liu, W., et al. *MicroMix: Efficient mixed-precision quantization with microscaling formats for large language models.* arXiv preprint arXiv:2508.02343, 2025.

[6] Jain, S. R., et al. *Trained quantization thresholds for accurate and efficient fixed-point inference of deep neural networks.* 3rd Machine Learning and Systems (MLSys) Conference, 2020.

[7] Bengio, Y., Léonard, N., and Courville, A. *Estimating or propagating gradients through stochastic neurons for conditional computation.* arXiv preprint arXiv:1308.3432, 2013.
