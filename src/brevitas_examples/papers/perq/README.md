# Pushing the Limits of Block Rotations in Post-Training Quantization [ICML 2026]

📄 [Paper](https://openreview.net/pdf?id=nvehxSdMqg)
💻 [Code](https://github.com/Xilinx/brevitas/blob/dev/src/brevitas/graph/permute.py)
💡 [Docs](https://xilinx.github.io/brevitas/dev/papers/perq.html)

```bibtex
@inproceedings{sanjeet2026perq,
      title={Pushing the Limits of Block Rotations in Post-Training Quantization},
      author={Sai Sanjeet and Ian Colbert and Pablo Monteagudo-Lago and Giuseppe Franco and Yaman Umuroglu and Nicholas J. Fraser},
      booktitle={Forty-third International Conference on Machine Learning},
      year={2026},
      url={https://openreview.net/forum?id=nvehxSdMqg}
}
```

> [!IMPORTANT]
> These yaml files were tested with `transformers==4.57.3` and `lighteval==0.13.0`

## Configs

| Config | Description |
|---|---|
| `llama3-perq_star-int4.yml` | PeRQ\* — block rotations + MassDiff + Qronos, W4A4 |
| `llama3-perq_dag-int4.yml` | PeRQ† — learned rotations (CayleySGD) + MassDiff + RTN, W4A4 |
| `benchmark-rotation_block_size.yml` | Multi-GPU sweep over block sizes and permutation strategies |

The provided configurations specify `meta-llama/Llama-3.2-1B-Instruct` by default.

## Running

```bash
brevitas_ptq_llm --config llama3-perq_star-int4.yml
```

Override the model:

```bash
brevitas_ptq_llm --config llama3-perq_star-int4.yml --model meta-llama/Llama-3.2-3B-Instruct
```

Key flags:

- `--rotation-block-size` — block size `b` for online Hadamard rotations (e.g. `16`, `32`, `64`).
  Smaller blocks reduce online rotation cost but suppress outliers less effectively without
  PeRQ. Omit for full-vector rotations (permutations are not applied in that case).
- `--permute-fn` — permutation strategy (`massdiff`, `zigzag`, `absmax`, `random`).
  Omit or set to `null` to disable permutations.
- `--disable-block-rotation-for-fused` — use block rotations only for online (orphan-sink)
  rotations; keep fused rotations as full-vector. This is the setting used by PeRQ\*.

## Benchmarking

To sweep over multiple configurations in parallel across GPUs:

```bash
python benchmark.py --config benchmark-rotation_block_size.yml --results results/ --gpus 0,1
```

`--gpus` is a comma-separated list of GPU device indices (e.g. `0,1,2,3`). Each GPU runs one
experiment at a time; experiments are dispatched as GPUs become available.

Below are WikiText2 perplexity results sweeping block size on Llama-3.2-1B-Instruct (W4A4,
per-channel weights), comparing MassDiff permutations against no permutation. Collected with
`python==3.12` and `torch==2.6.0`; different versions may yield different results.

| Block Size  | 16   | 32   | 64   | 128  | 256  | 512  | Full |
|-------------|------|------|------|------|------|------|------|
| No Permute  | 35.9 | 26.5 | 22.9 | 20.4 | 19.1 | 17.3 | 16.2 |
| PeRQ        | 18.2 | 17.0 | 16.6 | 16.1 | 16.1 | 15.9 | 16.2 |

## Reproducing paper experiments

Please use [this branch](https://github.com/i-colbert/brevitas/tree/permutations/src/brevitas_examples/papers/perq)
to reproduce the experiments from the paper.
