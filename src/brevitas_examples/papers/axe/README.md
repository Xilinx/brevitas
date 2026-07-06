# Accumulator-Aware Post-Training Quantization for Large Language Models [TMLR 2025]

📄 [Paper](https://openreview.net/pdf?id=p6l0579yj7)
💻 [Code](https://xilinx.github.io/brevitas/dev/papers/axe.html)

```bibtex
@article{colbert2025accumulatoraware,
      title={Accumulator-Aware Post-Training Quantization for Large Language Models},
      author={Ian Colbert and Giuseppe Franco and Fabian Grob and Jinjie Zhang and Rayan Saab},
      journal={Transactions on Machine Learning Research},
      issn={2835-8856},
      year={2025},
      url={https://openreview.net/forum?id=p6l0579yj7}
}
```

## Configs

| Config | Description |
|---|---|
| `benchmark-llama3-channelwise.yml` | Sweep of AXE configurations with per-channel BF16 scales and layerwise Hadamard rotations |
| `benchmark-llama3-groupwise.yml` | Sweep of AXE configurations with per-group power-of-2 (PO2) scales and layerwise Hadamard rotations |

All configurations quantize weights to 4-bit integers and apply layerwise Hadamard rotations.

Key flags:

- `--gptq` / `--gpfq` — error correction algorithm; set exactly one.
- `--gpxq-max-accumulator-bit-width` — accumulator bit width target. When unset, no AXE
  constraint is applied (equivalent to an unconstrained accumulator).
- `--gpxq-max-accumulator-tile-size` — partitions the dot product into tiles, each with its
  own accumulator budget. Defaults to the full dot product (monolithic accumulator). For
  groupwise quantization, this must equal the weight group size.
- `--input-bit-width` — activation bit width; sweep over `4` and `8`.
- `--input-quant-type` — symmetric (`sym`) or asymmetric (`asym`) activation quantization.

> [!NOTE]
> The paper evaluates asymmetric activation quantization with per-channel weight scales. Symmetric activation quantization and per-group weight scales are supported extensions introduced in [#1181](https://github.com/Xilinx/brevitas/pull/1181).

## Benchmarking

```bash
python benchmark.py --config benchmark-llama3-channelwise.yml --results results/ --gpus 0,1
```

`--gpus` accepts a comma-separated list of GPU indices; one experiment runs per GPU at a time.

> [!IMPORTANT]
> These yaml files were tested with `torch==2.6.0`, `transformers==4.57.6`, and `lighteval==0.13.0`

Results are on `meta-llama/Llama-3.2-1B-Instruct`.
Config is `tile_size` × `accumulator_bit_width`;
`A4`/`A8` denote activation bit width;
perplexity is measured over the WikiText2 test dataset (PPL, lower is better);
zero-shot accuracies are evaluated using the lighteval harness (%, higher is better).

**BF16 baseline**

| Wiki2 | ARC-C | ARC-E | WinoG | HellaS | PIQA | Avg. |
|-------|-------|-------|-------|--------|------|------|
| 11.7 |  34.1 |  68.8 |  57.9 |   45.6 | 74.6 | 56.2 |

### Channelwise scaling

Per-channel BF16 scales for weights; dynamic per-row activations.

> [!NOTE]
> For asymmetric activations with M=N=4 and K=128, Equation 3 in the paper gives P\*=16 —
> meaning the 128×32b and 128×16b configs are equivalent under asymmetric activations (the
> 16b accumulator is already sufficient). The key comparison is therefore W4A8 128×16b vs
> W4A4 128×16b: accumulator-aware quantization with 8-bit inputs preserves more quality than
> naively reducing input precision to 4-bit.

**A2GPTQ**

| Precision | Config  | Act. Type | Wiki2 | ARC-C | ARC-E | WinoG | HellaS | PIQA | Avg. |
|-----------|---------|-----------|-------|-------|-------|-------|--------|------|------|
| W4A8      | 128×32b | asym      | 12.8  |  32.0 |  64.4 |  58.6 |   44.3 | 72.1 | 54.3 |
| W4A8      | 128×16b | asym      | 14.1  |  31.2 |  61.4 |  56.0 |   42.5 | 70.9 | 52.4 |
| W4A4      | 128×16b | asym      | 15.7  |  32.1 |  60.2 |  54.1 |   42.3 | 68.9 | 51.5 |
|           |         |           |       |       |       |       |        |      |      |
| W4A8      | 128×32b | sym       | 12.8  |  33.8 |  64.4 |  56.6 |   44.4 | 72.4 | 54.3 |
| W4A8      | 128×16b | sym       | 12.8  |  33.1 |  66.0 |  57.4 |   44.1 | 73.0 | 54.7 |
| W4A4      | 128×16b | sym       | 15.9  |  29.9 |  59.3 |  54.5 |   40.9 | 68.9 | 50.7 |

**A2GPFQ**

| Precision | Config  | Act. Type | Wiki2 | ARC-C | ARC-E | WinoG | HellaS | PIQA | Avg. |
|-----------|---------|-----------|-------|-------|-------|-------|--------|------|------|
| W4A8      | 128×32b | asym      | 12.8  |  31.4 |  63.9 |  56.2 |   43.6 | 73.0 | 53.6 |
| W4A8      | 128×16b | asym      | 13.8  |  32.7 |  63.6 |  55.8 |   42.0 | 71.8 | 53.2 |
| W4A4      | 128×16b | asym      | 16.0  |  30.3 |  61.7 |  55.9 |   39.9 | 68.5 | 51.3 |
|           |         |           |       |       |       |       |        |      |      |
| W4A8      | 128×32b | sym       | 12.8  |  31.7 |  65.8 |  57.3 |   44.0 | 72.0 | 54.1 |
| W4A8      | 128×16b | sym       | 12.9  |  31.9 |  64.3 |  55.1 |   43.7 | 72.1 | 53.4 |
| W4A4      | 128×16b | sym       | 15.8  |  29.0 |  59.8 |  53.5 |   39.9 | 68.7 | 50.2 |

### Groupwise scaling

Per-group scales for weights; symmetric dynamic per-group activations (group size 32).
Scales are quantized to a power of 2 (PO2).

**A2GPTQ**

| Precision | Config | Wiki2 | ARC-C | ARC-E | WinoG | HellaS | PIQA | Avg. |
|-----------|--------|-------|-------|-------|-------|--------|------|------|
| W4A8      | 32×32b | 12.59 |  32.5 |  66.7 |  57.1 |   44.5 | 73.4 | 54.9 |
| W4A8      | 32×14b | 17.98 |  30.3 |  61.0 |  53.5 |   39.5 | 69.5 | 50.8 |
| W4A4      | 32×14b | 14.56 |  32.3 |  63.5 |  54.8 |   42.7 | 69.4 | 52.5 |

**A2GPFQ**

| Precision | Config | Wiki2 | ARC-C | ARC-E | WinoG | HellaS | PIQA | Avg. |
|-----------|--------|-------|-------|-------|-------|--------|------|------|
| W4A8      | 32×32b | 12.62 |  34.0 |  66.6 |  56.6 |   44.4 | 73.6 | 55.0 |
| W4A8      | 32×14b | 15.12 |  31.4 |  61.3 |  55.3 |   39.0 | 69.1 | 51.2 |
| W4A4      | 32×14b | 14.44 |  31.6 |  64.2 |  55.5 |   42.0 | 70.0 | 52.7 |
