=================================
Rotations in Brevitas
=================================

Why are rotations important?
----------------------------------------------------------

Large Language Models exhibit *computational invariance* [1]_, i.e. regions in which, if an invertible linear operation is applied to the output of a set of source modules and, conversely, its inverse is applied on the input of a set of sink modules, the output of the model remains unchanged (assuming sufficient precision). This invariance has been leveraged by applying random orthogonal transformations (whose inverse is its transpose) on the weights of the modules in these regions [2]_, which effectively removes weight and activation outliers, thus improving their amenability to quantization. Moreover, some of these rotations can be fused into the weights of the region's modules, so FP inference performance is not affected.

However, random orthogonal rotations generally improve quantization amenability in low-bit regimes. However, performance exhibits a large variance under different random rotations, as observed in [4]_. Consequently, these authors propose to further optimize the rotations, to improve quantized performance. In order to do so, they leverage the Cailey-SGD optimizer to ensure that the optimized rotations stay within the Stiefel manifold during optimization [5]_.


Rotations in Brevitas
----------------------------------------------------------

Brevitas enables to add rotations to an arbitrary model in a fined-grained manner through a number of options, specified in the LLM entrypoint (`brevitas_examples/llm/llm_args.py`):

- **--rotation** (*'fx', 'layerwise', 'fused_no_fx'*). If *'layerwise'*, each linear layer is wrapped in a `RotatedModule`, which rotates the input to the module by an orthogonal (Hadamard) matrix, while its inverse is fused into the weights of the linear layer. On the other hand, for 'fx' or 'fused_no_fx', Brevitas automatically detects the regions exhibiting rotation invariance, fusing the rotations into the weights of sources/sinks.
- **--rotation-mode** (*'had', 'ort'*). If *'had'*, random Hadamard matrices are used for rotations, which provide tighter bounds and are more efficient to apply [1]_. Therefore, this option is generally preferable to *'ort'*, which uses arbitrary random orthogonal matrices.
- **--rotation-orphan-sink**. If enabled, linear layers that are not sinks in any other rotation-invariant region are wrapped in a `RotatedModule`, as described for **--rotation** 'layerwise'.
- **--rotation-sdpa-regions**. If enabled, the value/output region (R₂ in [4]_) is rotated.

Moreover, similarly to [5]_, Brevitas can leverage the Cailey-SGD optimizer to further optimize the rotations, which can be enabled by setting the flag **--optimize-rotations**. The rotation training procedure relies on the `HF Trainer <https://huggingface.co/docs/transformers/en/main_classes/trainer>`_ class, and, therefore, can be configured by passing arguments accepted by the dataclass `TrainingArguments <https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments>`_. Moreover, the number of samples used for rotation calibration can be configured through the parameter **--nsamples-rot-calibration**.

Following, we provide a minimal example configuration for optimizing, in a single GPU, the rotations of a `HuggingfaceTB/SmolLM2-135M` model, with its weights quantized to 4 bits:

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

Note that the training parameters used in the SpinQuant paper [5]_ can be found in their `repository <https://github.com/facebookresearch/SpinQuant>`_.

Optimizing rotations in multiple GPUs
----------------------------------------------------------

As mentioned before, rotation optimization leverages the `HF Trainer <https://huggingface.co/docs/transformers/en/main_classes/trainer>`_ class. Therefore, to optimize rotations in a distributed environment, the LLM entrypoint has to be launched as an `accelerate script <https://huggingface.co/docs/accelerate/en/basic_tutorials/launch>`_ using the command `accelerate launch`.

To do so, the first step is to select the environment configuration through the command `accelerate config`, which provides an easy-to-use interface to specify the distributed environment. Once finished, a configuration file is generated, which can be passed to `accelerate launch` by setting the `--config_file` flag. Following, we provide an example configuration for a single-node environment with 2 GPUs:

.. code-block:: yaml

   compute_environment: LOCAL_MACHINE
   debug: false
   distributed_type: MULTI_GPU
   downcast_bf16: 'no'
   enable_cpu_affinity: false
   gpu_ids: 0,1
   machine_rank: 0
   main_training_function: main
   mixed_precision: 'no'
   num_machines: 1
   num_processes: 2
   rdzv_backend: static
   same_network: true
   tpu_env: []
   tpu_use_cluster: false
   tpu_use_sudo: false
   use_cpu: false

Once the configuration file is generated, the LLM entrypoint can be run in a distributed fashion as follows:

.. code-block:: shell

   accelerate launch --config_file ${configFolder}/accelerate_config.yaml ${workspaceFolder}/src/brevitas_examples/llm/main.py --config ${configFolder}/experiment_config.yaml 

Caveats
----------------------------------------------------------

Currently, we only support distributed training using `DistributedDataParallel`, and we plan to provide support for `DeepSpeed` and  `FullyShardedDataParallel` in the future. 

References
--------------------------------------------------

.. [1] Ashkboos, S., Croci, M. L., Nascimento, M. G. D., Hoefler, T., & Hensman, J. (2024). Slicegpt: Compress large language models by deleting rows and columns. arXiv preprint arXiv:2401.15024.
.. [2] Ashkboos, S., Mohtashami, A., Croci, M., Li, B., Cameron, P., Jaggi, M., ... & Hensman, J. (2025). Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37, 100213-100240.
.. [3] Tseng, A., Chee, J., Sun, Q., Kuleshov, V., & De Sa, C. (2024). Quip#: Even better llm quantization with hadamard incoherence and lattice codebooks. arXiv preprint arXiv:2402.04396.
.. [4] Liu, Z., Zhao, C., Fedorov, I., Soran, B., Choudhary, D., Krishnamoorthi, R., ... & Blankevoort, T. (2024). Spinquant: Llm quantization with learned rotations. arXiv preprint arXiv:2405.16406.
.. [5] Li, J., Fuxin, L., & Todorovic, S. (2020). Efficient riemannian optimization on the stiefel manifold via the cayley transform. arXiv preprint arXiv:2002.01113.