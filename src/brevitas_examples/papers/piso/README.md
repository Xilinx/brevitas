# Optimal Post-Training Quantization Scales and Where to Find Them

📄 [Paper](https://arxiv.org/abs/2606.10890)

```bibtex
@article{amboage2026piso,
  title={Optimal Post-Training Quantization Scales and Where to Find Them},
  author={Amboage, Juan and Monteagudo-Lago, Pablo and Colbert, Ian and Franco, Giuseppe and Fraser, Nicholas},
  journal={arXiv preprint arXiv:2606.10890},
  year={2026},
  url={https://arxiv.org/abs/2606.10890}
}
```

## Overview

PiSO (Piecewise Scale Optimization) computes the optimal weight quantization scale for a
data-aware layer-output reconstruction objective under round-to-nearest quantization. It
can be used standalone (decoupled from error correction) or interleaved with GPTQ / Qronos.

Scale-optimization arguments are exposed under the `--sopt-*` prefix; PiSO is the current
solver behind them.

## Configs

| Config | Description | Paper table |
|---|---|---|
| `benchmark-channelwise.yml` | Channel-wise weight-only quantization at 2/3/4-bit | Tables 1 and 2 |
| `benchmark-groupwise.yml` | Group-wise weight-only quantization at 2/3/4-bit, G16 and G32 | Table 3 |

Both files select `meta-llama/Llama-3.2-1B` and keep the paper's other models commented
out under `#### CHOOSE MODEL ####`; uncomment to extend the sweep.

## Benchmarking

```bash
python benchmark.py --config benchmark-channelwise.yml --results-folder results/ --gpus 0,1
```

`--gpus` accepts a comma-separated list of GPU indices; one experiment runs per GPU at a
time. Add `--dry-run` to print the resolved experiment list without running anything.

The BF16 reference of each table is reported as `float_ppl` in every run, so it does not
need a dedicated configuration.

## How the sweep maps to the paper rows

The configs sweep the scale-selection axis against the error-correction axis. Their
Cartesian product contains combinations that are not paper rows, so `benchmark.py`
overrides `validate` to discard them (`GridSearchUtils` drops any combination whose
validation raises). Each row below is therefore run exactly once.

### Scale selection

| Paper row | Flags |
|---|---|
| `+ absmax` | `sopt_optimize: false` (with `weight_param_method: stats`) |
| `+ data-free` | `sopt_optimize: true`, `sopt_hessian_mode: identity` |
| `+ PiSO` | `sopt_optimize: true`, `sopt_hessian_mode: dense` |
| `PiSO_I` / `PiSO_S` | `sopt_group_sequential: false` / `true` (group-wise only) |

### Integration with error correction

| Paper row | Flags |
|---|---|
| `RTN` | `gptq: false`, `qronos: false` |
| `+` (decoupled) | `sopt_optimize_in_gpxq: false` |
| `⊙ *` (layer-interleaved) | `sopt_optimize_in_gpxq: true`, `sopt_group_gpxq_layer_interleaved: true` |
| `⊙ †` (group-interleaved) | `sopt_optimize_in_gpxq: true`, `sopt_group_gpxq_layer_interleaved: false` |

Per-channel quantization is always layer-interleaved, so `⊙` needs only
`sopt_optimize_in_gpxq: true` there.

### Objective

PiSO minimizes the same reconstruction objective as the algorithm it is combined with, so
`sopt_objective` is not a free axis: `benchmark.py` keeps `self-activation` with `--gptq`
and `cross-activation` with `--qronos` or standalone RTN, and discards the rest.

### Discarded combinations

| Discarded | Reason |
|---|---|
| Any `sopt_objective` mismatching the error-correction algorithm | Not a paper configuration |
| `sopt_objective` variants when `sopt_optimize: false` | The objective is unused; deduplicated |
| `sopt_hessian_mode: identity` with `sopt_optimize: false` | Data-free requires scale optimization |
| `sopt_group_sequential` with `sopt_optimize: false`, `sopt_hessian_mode: identity`, or per-channel weights | The heuristic is inert; deduplicated |
| Layer-interleaved `sopt_hessian_mode: identity` | Reproduces the decoupled run (see below); deduplicated |

Interleaving the data-free baseline is only redundant in the *layer*-interleaved case.
The scale is then optimized in `_on_error_correction_starts`, before error correction
touches the layer, so `||w - s q(w)||^2` sees the original weights and returns the
decoupled scales. In the *group*-interleaved case the scale of each group is optimized in
`single_group_update` as GPTQ/Qronos reaches it, reading weights that already carry the
error diffused by the preceding groups, so `⊙ data-free †` is a genuinely distinct row and
is kept. Per-channel quantization has no group-interleaved mode, so data-free is always
deduplicated there.

### Resulting rows

Per model and bit width, `benchmark-channelwise.yml` yields 11 runs:

```
absmax:    RTN, GPTQ, Qronos
data-free: RTN, GPTQ, Qronos
PiSO:      RTN, GPTQ +, GPTQ ⊙, Qronos +, Qronos ⊙
```

Per model, bit width and group size, `benchmark-groupwise.yml` yields 22 runs:

```
absmax:    RTN, GPTQ, Qronos
data-free: RTN, {GPTQ, Qronos} +          (decoupled)
                {GPTQ, Qronos} ⊙†         (group-interleaved)
PiSO:      RTN + PiSO_{I,S}
           {GPTQ, Qronos} + PiSO_{I,S}    (decoupled)
           {GPTQ, Qronos} ⊙* PiSO_{I,S}   (layer-interleaved)
           {GPTQ, Qronos} ⊙† PiSO_{I,S}   (group-interleaved)
```

The nine GPTQ rows are exactly those of the extended group-wise table; the Qronos rows
mirror them.

Table 3 entries without an `I`/`S` subscript report the best of the two heuristics, so both
are run. Table 3 restricts the error-correction comparison to GPTQ; the Qronos rows above
are a superset and can be dropped by setting `qronos: [false]`.

> [!NOTE]
> **Beacon** (Tables 1 and 2) is not reproducible here: it has no Brevitas implementation
> and no public code release.

## Paper settings

Calibration uses 512 WikiText2 sequences of 2048 tokens; the paper reports 128, so set
`nsamples: [128]` to match it exactly. Perplexity is measured on WikiText2 and zero-shot
accuracy with LightEval. The paper averages ARC-Easy, ARC-Challenge and HellaSwag; the
configs additionally evaluate WinoGrande, which should be excluded from that average.
All scales are BF16 (`weight_scale_precision: float_scale`) and quantization is symmetric.
