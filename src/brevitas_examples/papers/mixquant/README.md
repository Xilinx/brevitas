# MixQuant: Pushing the Limits of Block Rotations in Post-Training Quantization

📄 [Paper](https://arxiv.org/pdf/2601.22347)
💻 [Code](https://github.com/Xilinx/brevitas/pull/1448)
💡 [Docs](https://xilinx.github.io/brevitas/dev/papers/mixquant.html)

```
@article{sanjeet2026mixquant,
      title={MixQuant: Pushing the Limits of Block Rotations in Post-Training Quantization},
      author={Sai Sanjeet and Ian Colbert and Pablo Monteagudo-Lago and Giuseppe Franco and Yaman Umuroglu and Nicholas J. Fraser},
      year={2026},
      eprint={2601.22347},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2601.22347},
}
```

> [!IMPORTANT]
> These yaml files were tested with transformers==4.57.3 and lighteval==0.13.0

The provided configurations specify Llama-3.2-1B-Instruct, but you can specify different Huggingface
models in the CLI args. For example:

```bash
   brevitas_ptq_llm --config llama3-mixquant_star-int4.yml --model meta-llama/Llama-3.2-3B-Instruct
```

You can use `benchmark.py` to run more experiments as follows:

```bash
python benchmark.py --config benchmark-rotation_block_size.yml --results results/ --gpus 0,1
```
where `--gpus` refers to how many gpus to use. If multiple GPUs are specified, each one will be used to run an individual experiment.

Please use https://github.com/i-colbert/brevitas/tree/mixquant/src/brevitas_examples/papers/mixquant to reproduce experiments used in the paper.
