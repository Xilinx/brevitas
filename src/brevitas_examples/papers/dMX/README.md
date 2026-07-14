# Learned Float Bit-Width with Rotation Optimisation

📄 [Paper](https://arxiv.org/abs/2606.04115)

This directory contains a custom quantizer and a custom trainer that jointly
optimise **rotation matrices** and **learned float bit-widths** for
mixed-precision LLM quantization, together with a benchmark runner and example
YAML configurations.

## Configs

| File | Description |
|---|---|
| `custom_trainer.py` | Registers the `rotation_learned_bitwidth` trainer into `TRAINER_REGISTRY`. Jointly optimises rotations (CaileySGD) and learned float bit-widths (SGD), with a bit-width regularisation penalty and temperature annealing. |
| `learned_float_quantizer.py` | Registers the `learned_float` and `mxfp6_learned_float` quantizers into `QUANTIZERS_REGISTRY`. |
| `benchmark.py` | Benchmark entrypoint. Importing it registers the quantizer and trainer as an import side effect, so they can be referenced by bare name in the YAML. It also raises the `torch._dynamo` recompilation limit. |
| `benchmark/mxfp8_mixed_precision.yaml` | MXFP8 mixed-precision sweep (`custom_quantizer: learned_float`). |
| `benchmark/mxfp6_mixed_precision.yaml` | MXFP6 mixed-precision sweep (`custom_quantizer: mxfp6_learned_float`). |
| `benchmark/mxfp8_big_models_mixed_precision.yaml` | MXFP8 sweep tuned for larger models (3B–8B): more calibration samples, more steps, smaller batch with gradient accumulation. |
| `benchmark/mxfp6_big_models_mixed_precision.yaml` | MXFP6 counterpart of the big-models sweep. |

All YAMLs sweep the `target_bit_width` regularisation target across several
models; the `mxfp8`/`mxfp6` pairs differ only in the quantizer, while the
`big_models` variants target larger models with adjusted training budget.

## Running

From this directory:

```bash
python benchmark.py --config benchmark/mxfp8_mixed_precision.yaml --results results/ --gpus 0,1
```

where `--gpus` is a comma-separated list of GPU device indices. Each argument
combination runs on its own process/GPU; with multiple GPUs, combinations run in
parallel.

Use `--dry-run` to print the expanded set of experiments without running them:

```bash
python benchmark.py --config benchmark/mxfp8_mixed_precision.yaml --dry-run
```

## How it works

The custom trainer and quantizer are plugins that register themselves into
Brevitas' registries when their module is imported. There are two ways to load
them:

1. **By plugin path** (`path/to/plugin.py:name`), handled directly by the LLM
   entrypoint via `--custom-trainer` / `--custom-quantizer`.
2. **By bare name**, which requires the plugin module to already be imported.

`benchmark.py` uses option (2): it imports `custom_trainer` and
`learned_float_quantizer` at module load, so the YAML can simply say
`custom_trainer: rotation_learned_bitwidth` and
`custom_quantizer: learned_float` without spelling out the full file path.

The benchmark entrypoint also sets:

```python
torch._dynamo.config.recompile_limit = 1000
```

Fine-tuning with the learned-float quantizer triggers many `torch.compile`
recompilations; the default limit is too low and would disable compilation
partway through, so it is raised.

## Configuration notes

The benchmark expands the YAML into the Cartesian product of all listed values
(each key maps to a list). Keys are routed automatically:

* Keys recognised by the LLM argument parser become top-level entrypoint
  arguments (e.g. `model`, `rotation`, `weight_bit_width`, `custom_trainer`,
  `custom_quantizer`).
* All other keys are forwarded as **training arguments** to the trainer's
  training-arguments class (`RotationLearnedBitWidthTrainingArguments`),
  e.g. `max_steps`, `per_device_train_batch_size`, `lr_scheduler_type`.

Fine-tuning is enabled with `fine_tune: true` (the flag `optimize_rotations` is
a deprecated alias). Selecting `--custom-trainer` implies `--fine-tune`.

Key training arguments exposed by `rotation_learned_bitwidth`:

| Argument | Meaning |
| --- | --- |
| `target_bit_width` | Target average bit-width for the regularisation penalty. |
| `rotation_lr` | Learning rate for CaileySGD (rotation matrices). |
| `bw_learning_rate` | Learning rate for SGD on the bit-width parameters. |
| `delay_start` | Fraction of `max_steps` to wait before starting temperature annealing. |
| `optimizer_dtype` | Dtype for the CaileySGD computations (e.g. `float32`). |
| `simple_average_loss` | If `true`, use the simple (unweighted) average bit-width for the loss; otherwise weight by tensor size. |

In the example YAMLs, `target_bit_width` is swept over a range of targets to
trace the accuracy/bit-width trade-off, while everything else is held fixed.
