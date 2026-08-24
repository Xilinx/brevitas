=================================
Rotations in Brevitas
=================================

Why are rotations important?
----------------------------------------------------------

Large Language Models exhibit *computational invariance* [1]_ meaning that applying an invertible
linear operation to the output of certain modules (sources), and its inverse to the input of others
(sinks), leaves the model's output unchanged (assuming sufficient precision).
This property allows for the selective application of random orthogonal transformations,
which effectively mitigate weight and activation outliers, enhancing their quantization amenability [2]_.
Moreover, some rotations can be fused into the module weights, thus preserving floating-point inference performance.

Although random orthogonal rotations generally improve quantization amenability in low-bit regimes,
the quantized network performance  exhibits a large variance under different random rotations,
as observed in [3]_. Consequently, these authors propose to further optimize the rotations to
improve quantized performance. In order to do so, they leverage the Cailey-SGD optimizer to ensure
that the optimized rotations stay within the Stiefel manifold during optimization [4]_.


Rotations in Brevitas
----------------------------------------------------------

In Brevitas, we support different flavours of rotations:

* *Layerwise* Rotations
* *Fused* Rotations
* *Fused Optimized* Rotations



Layerwise Rotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to merge rotations into subsequent and preceding layers, it is often required to apply certain
transformations to the network, such as converting LayerNorm to RMSNorm, as well as identifying the
suitable layers (or regions) in the network where such operation is doable.

This comes with some drawbacks, such as the impact on quantization of merging RMSNorm affine parameters,
and with some limitations.
In order to detect the regions of the model where it is possible to fuse rotations, we rely on FX graph,
in particular the one that can be extracted with `torch.dynamo._export`.

This API allows to get a modular FX graph, which means that calls to modules (i.e., Linear module) are
not flattened into their respective functional calls.

The limitation of this API resides in the fact that graph breaks are not allowed, which means that a
the entire network must contain only operators that can be traced through torch.dynamo.
Although compatibility has greatly improved in the past months, custom networks and operators might still
not be supported.


For all these reasons, we offer the possibility of applying rotations without having to worry about
merging the rotations in the preceding layers. From our experiments, this provides generally better
accuracy compared to the case where rotations are fused, although the computational price to pay is
generally higher.

In its most simple form, this can be done in the following way:

.. code-block:: python
    
    from brevitas.graph.equalize import LayerwiseActivationRotation

    eq = LayerwiseActivationRotation()
    model = eq.apply(model)

Fused Rotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following more closely the approach used in [1]_ and [2]_, it is possible to fuse rotations into 
preceding layers if the network structure allows it.

As mentioned, there are few assumptions in this case:

* The class accepts a FX-traced module (e.g., through `torch.dynamo._export`)
* LayerNorm have been transformed to RMSNorm if necessary
* RMSNorm affine parameters are merged into adjecent layers

Brevitas offers utilities to facilitate these transformations, as well as a standalone class for
FX-based rotation, called `GraphRotationEqualization`.

The most important options for the class are:

* orphan_sink: Whether to add standalone, unmerged rotations for the regions that do not support fusion
* full_rotation_method: Allows to select between Hadamard ('had') or orthogonal ('ort') rotations
* sdpa_regions: detect where torch.nn.functional.scaled_dot_product is placed in the network and appropriately applies rotations in the layers around it.
* use_parametrized_rotations: Register rotations are parametrization, allowing for a subsequent optimization, for example through Caley SGD.

It is important to note that *sdpa_regions* makes some assumptions about the structure of the attention
block, with the presence of QKV linear layers before, and an Output linea layer afterwards.

.. code-block:: python

    from brevitas.graph.equalize import LayerNormToRMS
    from brevitas.graph.equalize import MergeLnAffine
    from brevitas.graph.equalize import GraphRotationEqualization
    import torch
    
    with torch.no_grad():
        fx_model, guards = torch._dynamo.export(model)(example_input)
    
        # Merge the learned (affine) LayerNorm parameters
        eq = MergeLnAffine()
        fx_model = eq.apply(fx_model)

        # Convert LayerNorm to RMSNorm
        eq = LayerNormToRMS()
        fx_model = eq.apply(fx_model)

        eq = GraphRotationEqualization(
            orphan_sink=True
            full_rotation_method='had',
            sdpa_regions=True,
            use_parametrized_rotations=False)
        fx_model = eq.apply(fx_model)


If `use_parametrized_rotations` is False, this resembles closely the optimization structure presented in 
QuaRot [2]_.


Optimized Fused Rotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If `use_parametrized_rotations` is True, the rotations are registered as parametrizations ([5]_),
and as a parameters.
After adding the quantization blocks to the network, it is thus necessary to optimize the rotations and
then fuse them into the corresponding weights.

Brevitas offers an example of how to accomplish that in our LLM entrypoint (`brevitas_ptq_llm`).

In particular, there are two ways to apply optimized fused rotations through this entrypoint, `FX` and `FUSED_NO_FX`.

Although an FX representation of the model is fundamental for this process, it is possible to discard the FX representation
once the correct regions have been identified, and the transformations can be then applied on the
non-FX model.

The advantage of this approach is that we do not lose the hierarchical structure of the original model,
which is a side effect of `torch.dynamo._export`.
Maintaining this hierarchical structure allows us to optimize the execution of certain PTQ algorithms like
GPTQ or Qronos, by optimizing one block at a time and caching intermediate representations.


Moreover, similarly to [3]_, Brevitas can leverage the Cailey-SGD optimizer to further optimize the rotations.
The rotation training procedure relies on the
`HF Trainer <https://huggingface.co/docs/transformers/en/main_classes/trainer>`_ class, and, therefore,
can be configured by passing arguments accepted by the dataclass
`TrainingArguments <https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments>`_.
Moreover, the number of samples used for rotation calibration can be configured through the parameter ``--nsamples-rot-calibration``.

Following, we provide a minimal example configuration for optimizing, in a single GPU, the rotations
of a ``HuggingfaceTB/SmolLM2-135M`` model, with its weights quantized to 4 bits:

.. code-block:: yaml

   dataset: wikitext2
   eval: true
   model: HuggingfaceTB/SmolLM2-135M
   rotation: fused_no_fx
   optimize_rotations: true
   nsamples_rot_calibration: 800
   replace_rmsnorm: true
   weight_bit_width: 4
   dtype: float32
   learning_rate: 1.5
   weight_decay: 0.0
   lr_scheduler_type: cosine
   max_steps: 100
   per_device_train_batch_size: 2
   gradient_accumulation_steps: 4
   save_safetensors: false
   logging_steps: 10
   log_on_each_node: false


Note that the training parameters used in the SpinQuant paper [3]_ can be found in their `repository <https://github.com/facebookresearch/SpinQuant>`_.

.. rubric:: References

.. [1] Ashkboos, S., Croci, M. L., Nascimento, M. G. D., Hoefler, T., & Hensman, J. (2024). Slicegpt: Compress large language models by deleting rows and columns. arXiv preprint arXiv:2401.15024.
.. [2] Ashkboos, S., Mohtashami, A., Croci, M., Li, B., Cameron, P., Jaggi, M., ... & Hensman, J. (2025). Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37, 100213-100240.
.. [3] Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Krishnamoorthi, R., ... & Blankevoort, T. (2024). Spinquant: Llm quantization with learned rotations. arXiv preprint arXiv:2405.16406.
.. [4] Li, J., Fuxin, L., & Todorovic, S. (2020). Efficient riemannian optimization on the stiefel manifold via the cayley transform. arXiv preprint arXiv:2002.01113.
.. [5] https://docs.pytorch.org/tutorials/intermediate/parametrizations.html