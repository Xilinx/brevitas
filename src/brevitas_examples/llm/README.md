# LLM quantization

## Requirements

The requirements can be installed as followed:

```bash
pip install -e .[export,llm]
```

## Run

Set the env variable `BREVITAS_JIT=1` to speed up the quantization process. Currently unsupported whenever export is also toggled or with MSE based scales/zero-points.

When using `--optimize-rotations`, the rotation training procedure relies on the Trainer class (https://huggingface.co/docs/transformers/en/main_classes/trainer). Therefore, training can be further configured by passing arguments accepted by the dataclass TrainingArguments (https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments), e.g. `--learning_rate`, `--weight_decay`, `per_device_train_batch_size`.

```bash
usage: main.py [-h] [--config CONFIG] [--model MODEL] [--custom-quantizer CUSTOM_QUANTIZER] [--dtype {auto,float32,float16,bfloat16}] [--seed SEED] [--nsamples NSAMPLES]
               [--nsamples-rot-calibration NSAMPLES_ROT_CALIBRATION] [--seqlen SEQLEN] [--eval] [--eval-top-k EVAL_TOP_K] [--dataset {wikitext2,c4,pile,fineweb}] [--dataset-eval-split {validation,test}]
               [--gpxq-block-name GPXQ_BLOCK_NAME] [--gpxq-buffer-device {cpu,same}] [--weight-bit-width WEIGHT_BIT_WIDTH] [--weight-param-method {stats,mse,hqo}]
               [--weight-scale-precision {signed_float_scale,float_scale,po2_scale}] [--weight-quant-rescaling-init WEIGHT_QUANT_RESCALING_INIT] [--weight-narrow-range] [--weight-quant-type {sym,asym}]
               [--weight-quant-format WEIGHT_QUANT_FORMAT] [--weight-quant-granularity {per_channel,per_tensor,per_group}] [--scale-rounding-func-type {round,ceil,floor,midmax}]
               [--weight-group-dim {1,0}] [--weight-group-size WEIGHT_GROUP_SIZE] [--quantize-weight-zero-point] [--input-bit-width INPUT_BIT_WIDTH] [--input-quant-format INPUT_QUANT_FORMAT]
               [--input-param-method {stats,mse}] [--input-scale-precision {signed_float_scale,float_scale,po2_scale}] [--input-scale-type {static,dynamic,no_scale}] [--input-quant-type {sym,asym}]
               [--input-narrow-range] [--input-quant-granularity {per_tensor,per_row,per_group}] [--input-group-size INPUT_GROUP_SIZE] [--attn-quant-config {qkvs,qkv,kv}]
               [--attn-bit-width ATTN_BIT_WIDTH] [--attn-quant-format ATTN_QUANT_FORMAT] [--attn-param-method {stats,mse}] [--attn-scale-precision {signed_float_scale,float_scale,po2_scale}]
               [--attn-scale-type {static,dynamic,no_scale}] [--attn-quant-type {sym,asym}] [--attn-quant-granularity {per_tensor,per_row,per_group}] [--attn-group-size ATTN_GROUP_SIZE]
               [--learned-round-lr LEARNED_ROUND_LR] [--learned-round-scale-lr LEARNED_ROUND_SCALE_LR] [--learned-round-scale-momentum LEARNED_ROUND_SCALE_MOMENTUM]
               [--learned-round-iters LEARNED_ROUND_ITERS] [--learned-round-scale] [--quantize-input-zero-point] [--quantize-first-last-layer] [--magr] [--magr-alpha MAGR_ALPHA] [--qronos]
               [--qronos-alpha QRONOS_ALPHA] [--gptq] [--gpfq] [--gpxq-act-order] [--gpxq-use-quant-activations] [--gpxq-use-grid-optimization] [--disable-create-weight-orig]
               [--gpxq-max-accumulator-bit-width GPXQ_MAX_ACCUMULATOR_BIT_WIDTH] [--gpxq-max-accumulator-tile-size GPXQ_MAX_ACCUMULATOR_TILE_SIZE] [--act-calibration] [--weight-calibration] [--bias-corr] [--ln-affine-merge]
               [--convert-layernorm-to-rmsnorm] [--replace-rmsnorm] [--no-quantize] [--scaling-min-val SCALING_MIN_VAL] [--quant-sdpa {eager,functional,fx}]
               [--eager-quant-sdpa-class EAGER_QUANT_SDPA_CLASS] [--weight-equalization] [--rotation {fx,layerwise,fused_no_fx}] [--fine-tune] [--optimize-rotations] [--custom-trainer CUSTOM_TRAINER]
               [--rotation-mode {had,ort}] [--rotation-orphan-sink] [--rotation-sdpa-regions] [--rotation-layers-to-expand [ROTATION_LAYERS_TO_EXPAND ...]] [--expansion-step EXPANSION_STEP]
               [--rotation-block-size ROTATION_BLOCK_SIZE] [--disable-block-rotation-for-fused] [--svd-quant] [--svd-quant-rank SVD_QUANT_RANK] [--svd-quant-iters SVD_QUANT_ITERS]
               [--act-equalization {None,layerwise,fx}] [--act-equalization-alpha ACT_EQUALIZATION_ALPHA] [--permute-fn {absmax,massdiff,zigzag,random}] [--export-target EXPORT_TARGET]
               [--export-prefix EXPORT_PREFIX] [--export-path EXPORT_PATH] [--checkpoint-name CHECKPOINT_NAME] [--load-checkpoint] [--learned-round {None,identity}] [--learned-round-fast-update]
               [--few-shot-eval {lm_eval,lighteval}] [--few-shot-override-batch-size FEW_SHOT_OVERRIDE_BATCH_SIZE] [--compile-ptq] [--compile-eval] [--few-shot-zeroshot]
               [--bos-preprocessing {None,document,sequence}] [--few-shot-limit FEW_SHOT_LIMIT] [--few-shot-tasks [FEW_SHOT_TASKS ...]] [--awq-scale] [--awq-clip]
               [--calibration-batch-size CALIBRATION_BATCH_SIZE] [--sopt-solver {piso,chunked-piso}]
               [--sopt-use-quant-activations | --no-sopt-use-quant-activations] [--sopt-solver-batch-size SOPT_SOLVER_BATCH_SIZE] [--sopt-quantize-prev | --no-sopt-quantize-prev]
               [--sopt-cross-act-objective | --no-sopt-cross-act-objective]
               [--sopt-group-sequential | --no-sopt-group-sequential] [--sopt-group-sequential-act-order | --no-sopt-group-sequential-act-order] [--sopt-hessian-mode {dense,diagonal,identity}]
               [--sopt-group-gpxq-layer-interleaved | --no-sopt-group-gpxq-layer-interleaved]

LLM quantization

options:
  -h, --help            show this help message and exit
  --config CONFIG       Specify alternative default commandline args (e.g., config/default_template.yml). Default: None.
  --model MODEL         HF model name. Default: facebook/opt-125m.
  --custom-quantizer CUSTOM_QUANTIZER
                        Override the quantization list and/or post-process the quantized model with a user-defined quantization plugin. The plugin can be a registered name or a .py file path followed by
                        :plugin_name. Default: None.
  --dtype {auto,float32,float16,bfloat16}
                        Data type for model. Default: auto
  --seed SEED           Seed for sampling the calibration data. Default: 0.
  --nsamples NSAMPLES   Number of calibration data samples. Set to -1 to load the entire dataset. Default: 128.
  --nsamples-rot-calibration NSAMPLES_ROT_CALIBRATION
                        Number of calibration data samples for rotation. Default: 800.
  --seqlen SEQLEN       Sequence length. Default: 2048.
  --eval                Eval model PPL, EAR, and KLD on the chosen dataset.
  --eval-top-k EVAL_TOP_K
                        Number of tokens to evaluate EAR and KLD on. Default: 10.
  --dataset {wikitext2,c4,pile,fineweb}
                        Dataset to use for quantization (default: wikitext2)
  --dataset-eval-split {validation,test}
                        Specify which split to use for the evaluation dataset (default: validation)
  --gpxq-block-name GPXQ_BLOCK_NAME, --block-name GPXQ_BLOCK_NAME
                        Attribute for model blocks. Used for faster GPxQ optimization (if FX is not needed) and learned round (default: None)
  --gpxq-buffer-device {cpu,same}
                        Device for GPxQ buffers. "same" means using the same device for the buffer as the layer weights. Default cpu
  --weight-bit-width WEIGHT_BIT_WIDTH
                        Weight bit width. Default: 8.
  --weight-param-method {stats,mse,hqo}
                        How scales/zero-point are determined. Default: stats.
  --weight-scale-precision {signed_float_scale,float_scale,po2_scale}
                        Whether scale is a float value or a po2. Default: float_scale.
  --weight-quant-rescaling-init WEIGHT_QUANT_RESCALING_INIT
                        Value to rescale weight quantization grid. Default: None (no rescaling).
  --weight-narrow-range
                        Use narrow range for weight quantization. Default: False.
  --weight-quant-type {sym,asym}
                        Weight quantization type. Default: asym.
  --weight-quant-format WEIGHT_QUANT_FORMAT
                        Weight quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It's possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.
  --weight-quant-granularity {per_channel,per_tensor,per_group}
                        Granularity for scales/zero-point of weights. Default: per_group.
  --scale-rounding-func-type {round,ceil,floor,midmax}
                        Rounding function to use with Po2 scale. Default: None.
  --weight-group-dim {1,0}
                        Override default group_dim for groupsize quantization. Default: layer-dependant
  --weight-group-size WEIGHT_GROUP_SIZE
                        Group size for per_group weight quantization. Default: 128.
  --quantize-weight-zero-point
                        Quantize weight zero-point.
  --input-bit-width INPUT_BIT_WIDTH
                        Input bit width. Default: None (disables input quantization).
  --input-quant-format INPUT_QUANT_FORMAT
                        Input quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It's possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: int.
  --input-param-method {stats,mse}
                        How scales/zero-point are determined. Default: stats (percentile for static, absmax or minmax for dynamic).
  --input-scale-precision {signed_float_scale,float_scale,po2_scale}
                        Whether input scale is a float value or a po2. Default: float.
  --input-scale-type {static,dynamic,no_scale}
                        Whether input scale is a static value or a dynamic value.
  --input-quant-type {sym,asym}
                        Input quantization type. Default: asym.
  --input-narrow-range  Use narrow range for input quantization. Default: False.
  --input-quant-granularity {per_tensor,per_row,per_group}
                        Granularity for scales/zero-point of inputs. Default: per_tensor.
  --input-group-size INPUT_GROUP_SIZE
                        Group size for per_group input quantization. Default: 64.
  --attn-quant-config {qkvs,qkv,kv}
                        Decide which parts of attention should be quantized. "kv" will only quantize KV, "qkvs" will quantize all MatMuls in attention (QKV & Softmax output). Note: --quant-sdpa needs be
                        set for this to have an effect. Default: qkvs
  --attn-bit-width ATTN_BIT_WIDTH
                        Attention bit width. Default: None (same as input).
  --attn-quant-format ATTN_QUANT_FORMAT
                        Attention quantization type. Either int or eXmY, with X+Y==weight_bit_width-1. It's possible to add float_ocp_ or float_fnuz_ before the exponent/mantissa bitwidth. Default: None
                        (same as input).
  --attn-param-method {stats,mse}
                        How scales/zero-point are determined. Default: None (same as input).
  --attn-scale-precision {signed_float_scale,float_scale,po2_scale}
                        Whether input scale is a float value or a po2. Default: (same as input).
  --attn-scale-type {static,dynamic,no_scale}
                        Whether input scale is a static value or a dynamic value. Default: (same as input).
  --attn-quant-type {sym,asym}
                        Input quantization type. Default: (same as input).
  --attn-quant-granularity {per_tensor,per_row,per_group}
                        Granularity for scales/zero-point of inputs. Default: (same as input).
  --attn-group-size ATTN_GROUP_SIZE
                        Group size for per_group input quantization. Default: (same as input).
  --learned-round-lr LEARNED_ROUND_LR
                        Learning rate for learned round parameter optimization. Default: 0.005
  --learned-round-scale-lr LEARNED_ROUND_SCALE_LR
                        Learning rate for scale optimization during round learning. Default: 0.01
  --learned-round-scale-momentum LEARNED_ROUND_SCALE_MOMENTUM
                        Learning rate for scale optimization during round learning. Default: 0.9
  --learned-round-iters LEARNED_ROUND_ITERS
                        Number of iterations for learned round. Default: 200.
  --learned-round-scale
                        Learned scale factor together with round.
  --quantize-input-zero-point
                        Quantize input zero-point.
  --quantize-first-last-layer
                        Quantize the first (embedding) and last nn.Linear layer. When not set, the embedding is left unquantized and the last layer weight quantizer is forced to None, overriding any
                        definition from a custom quantizer.
  --magr                Apply MagR.
  --magr-alpha MAGR_ALPHA
                        Alpha for MagR. Default: 0.01.
  --qronos              Apply Qronos.
  --qronos-alpha QRONOS_ALPHA
                        Alpha for Qronos. Default: 1e-6
  --gptq                Apply GPTQ.
  --gpfq                Apply GPFQ.
  --gpxq-act-order      Apply GPxQ activation ordering.
  --gpxq-use-quant-activations
                        Use quantized activations in GPxQ.
  --gpxq-use-grid-optimization
                        Use grid optimization with GPTQ or Qronos.
  --disable-create-weight-orig
                        Disable maintaining original weights for non-quant forward pass. Default: false
  --gpxq-max-accumulator-bit-width GPXQ_MAX_ACCUMULATOR_BIT_WIDTH
                        Maximum accumulator bit width for GPxQ using AXE.
  --gpxq-max-accumulator-tile-size GPXQ_MAX_ACCUMULATOR_TILE_SIZE
                        Maximum accumulator tile size for GPxQ using AXE.
  --act-calibration     Apply activation calibration.
  --weight-calibration  Apply weight calibration.
  --bias-corr           Apply bias correction.
  --ln-affine-merge     Merge LN affine params.
  --convert-layernorm-to-rmsnorm
                        Merge LN affine params.
  --replace-rmsnorm     Replace HF RMSNorms with Torch one.
  --no-quantize         Disable quantization.
  --scaling-min-val SCALING_MIN_VAL
                        Minimum value to clamp scale to when using bf16 or fp16 quantization.
  --quant-sdpa {eager,functional,fx}
                        Define how to quantize SDPA. (default: None)
  --eager-quant-sdpa-class EAGER_QUANT_SDPA_CLASS
                        If quant_sdpa is eager, specify the name of the attention class. (default: auto)
  --weight-equalization
                        Apply weight equalization. Relevant to ReLU based models (e.g. OPT).
  --rotation {fx,layerwise,fused_no_fx}
                        Apply graph rotation equalization
  --fine-tune           Enable the training / fine-tuning pipeline. When used with --rotation, rotation matrices are parametrized and optimized. When used without --rotation, a custom trainer plugin can
                        fine-tune all model weights (default: False).
  --optimize-rotations  Deprecated: use --fine-tune instead. Implies --fine-tune (default: False).
  --custom-trainer CUSTOM_TRAINER
                        Path to a custom training plugin, in the format "path/to/plugin.py:config_name". The plugin .py file should register a custom Trainer class into the TRAINER_REGISTRY. The Trainer
                        class may expose a "training_args_cls" class attribute to customise the training arguments (including the optimizer setup via "optimizer_scheduler_args"); otherwise the LLM
                        example defaults are used. Implies --fine-tune. Default: None.
  --rotation-mode {had,ort}
                        If GraphRotation is enabled, decide how to compute the random rotation matrix that is fully fused. Online or partial rotation will always be Hadamard
  --rotation-orphan-sink
                        If GraphRotation is enabled, decide wheter to add standalone hadamard matrices for the unfused layers
  --rotation-sdpa-regions
                        If GraphRotation is enabled, decide wheter to equalize across SDPA
  --rotation-layers-to-expand [ROTATION_LAYERS_TO_EXPAND ...]
                        A list of module names to expand with hadamard rotation. Default: []
  --expansion-step EXPANSION_STEP
                        When layer expansion is set, decide how much to increase the layer sizes. Default: 1
  --rotation-block-size ROTATION_BLOCK_SIZE
                        Perform blockwise rotations when possible. Default: None
  --disable-block-rotation-for-fused
                        Disable block rotations when using fused rotations. Default: False
  --svd-quant           Apply SVDQuant.
  --svd-quant-rank SVD_QUANT_RANK
                        Rank to use for SVDQuant (default: 32).
  --svd-quant-iters SVD_QUANT_ITERS
                        Number of iterations to use for SVDQuant (default: 1).
  --act-equalization {None,layerwise,fx}
                        Apply activation equalization (SmoothQuant). Layerwise introduces standalone mul nodes,while fx merges them whenever possible into previous tensors, which is possible on ReLU
                        based models (e.g. OPT).
  --act-equalization-alpha ACT_EQUALIZATION_ALPHA
                        If activation equalization is enabled, decide what alpha to use
  --permute-fn {absmax,massdiff,zigzag,random}
                        Permutation function to use. If None, no permutation is applied. Works with block rotations when both are enabled.
  --export-target EXPORT_TARGET
                        Model export. One of 'vllm', 'shark', 'onnx_qcdq', 'sharded_torchmlir_group_weight', 'sharded_packed_torchmlir_group_weight', or 'gguf:<ftype>' (e.g. 'gguf:q4_k_s'; see
                        brevitas_examples.llm.gguf_export.targets.GGUF_EXPORT_TARGETS for the full list of supported ftypes). Default: None
  --export-prefix EXPORT_PREFIX
                        Path prefix to use for the various export flows. If None, a path will be derived from the model name (default: None)
  --export-path EXPORT_PATH
                        Output path for GGUF export. If it ends in '.gguf', treated as the exact file to write (parent directories created as needed); otherwise treated as a directory to write into using
                        gguf-py's auto-derived filename. If None, writes to the current working directory. Default: None
  --checkpoint-name CHECKPOINT_NAME
                        Filename to save checkpoint. If `None`, no checkpoint is saved (default: None)
  --load-checkpoint     Boolean flag to load_checkpoint, uses checkpoint_name. Default False)
  --learned-round {None,identity}
                        Whether to use learned round. If `None`, RTN is used (default: None)
  --learned-round-fast-update
                        Whether to use fast update with learned round. Prototype (default: False)
  --few-shot-eval {lm_eval,lighteval}
                        Perform zero_shot evaluation with lm_eval or lighteval. Default None)
  --few-shot-override-batch-size FEW_SHOT_OVERRIDE_BATCH_SIZE
  --compile-ptq         Compile for PTQ algorithms. Default False)
  --compile-eval        Compile during evaluation. Default False)
  --few-shot-zeroshot   Whether to do zero or few shot eval. Default False)
  --bos-preprocessing {None,document,sequence}
                        Type of BOS token pre-processing for training and evaluation datasets. Default None)
  --few-shot-limit FEW_SHOT_LIMIT
                        Few shot limit. Default None)
  --few-shot-tasks [FEW_SHOT_TASKS ...]
                        A list of tasks for zero_shot evaluation. Default: ['arc_challenge', 'arc_easy', 'winogrande', 'piqa']
  --awq-scale           Whether to apply AWQ scaling (default: False).
  --awq-clip            Whether to apply AWQ clipping (default: False).
  --calibration-batch-size CALIBRATION_BATCH_SIZE
                        Batch size for calibration data loader. (default: 1).
  --sopt-solver {piso,chunked-piso}
                        Scale optimization solver family. chunked-piso is a faster per-channel variant that only supports positive scales (symmetric grids). Default: piso
  --sopt-use-quant-activations
                        Enable Use quantized activations during weight calibration. Default: enabled
  --no-sopt-use-quant-activations
                        Disable Use quantized activations during weight calibration. Default: enabled
  --sopt-solver-batch-size SOPT_SOLVER_BATCH_SIZE
                        The number of weight rows/channels in a given layer for which the scale is optimized in parallel. When set to None all the scales in the layer are optimized at the same time
                        (fastest but more memory required).
  --sopt-quantize-prev
                        Enable Use quantized activations to reconstruct the current layer. Default: disabled
  --no-sopt-quantize-prev
                        Disable Use quantized activations to reconstruct the current layer. Default: disabled
  --sopt-cross-act-objective
                        Enable Use float activations as the reconstruction target. Requires --sopt-quantize-prev. Default: disabled
  --no-sopt-cross-act-objective
                        Disable Use float activations as the reconstruction target. Requires --sopt-quantize-prev. Default: disabled
  --sopt-group-sequential
                        Enable With per-group weights, optimize each group while correcting the error from previous groups (sequential heuristic). When disabled, groups are optimized independently.
                        Default: disabled
  --no-sopt-group-sequential
                        Disable With per-group weights, optimize each group while correcting the error from previous groups (sequential heuristic). When disabled, groups are optimized independently.
                        Default: disabled
  --sopt-group-sequential-act-order
                        Enable With --sopt-group-sequential during weight calibration, process groups by descending sum of diag(H) (most important first) instead of natural order. Ignored with --gpxq-
                        use-grid-optimization, where the column order is controlled by --gpxq-act-order. Default: enabled
  --no-sopt-group-sequential-act-order
                        Disable With --sopt-group-sequential during weight calibration, process groups by descending sum of diag(H) (most important first) instead of natural order. Ignored with --gpxq-
                        use-grid-optimization, where the column order is controlled by --gpxq-act-order. Default: enabled
  --sopt-hessian-mode {dense,diagonal,identity}
                        How the solver uses the Hessian H and matrix G. dense: the full matrices accumulated from the calibration data; not implemented, so it currently raises NotImplementedError.
                        diagonal: only diag(H) (and diag(G)), discarding cross-weight interactions. identity: ignore the calibration data entirely and use H=I, G=None (pure weight MSE). Default: dense
  --sopt-group-gpxq-layer-interleaved
                        Enable Selects how the solver is interleaved with GPxQ for group-wise weight quantization (paper: interleaved integration strategies). When enabled, all group scales are optimized
                        once per layer (layer-interleaved). When disabled (default), each group scale is optimized at its boundary during GPxQ (group-interleaved). Only applies with --gpxq-use-grid-
                        optimization and per-group weights; per-channel quantization is always layer-interleaved regardless. Default: disabled
  --no-sopt-group-gpxq-layer-interleaved
                        Disable Selects how the solver is interleaved with GPxQ for group-wise weight quantization (paper: interleaved integration strategies). When enabled, all group scales are
                        optimized once per layer (layer-interleaved). When disabled (default), each group scale is optimized at its boundary during GPxQ (group-interleaved). Only applies with --gpxq-
                        use-grid-optimization and per-group weights; per-channel quantization is always layer-interleaved regardless. Default: disabled
```
