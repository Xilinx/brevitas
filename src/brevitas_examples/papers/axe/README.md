# Accumulator-Aware Post-Training Quantization for Large Language Models

📄 [Paper](https://openreview.net/forum?id=p6l0579yj7)

```
@article{colbert2025accumulatoraware,
  title={Accumulator-Aware Post-Training Quantization for Large Language Models},
  author={Ian Colbert and Giuseppe Franco and Fabian Grob and Jinjie Zhang and Rayan Saab},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=p6l0579yj7},
  note={}
}
```

Please use `benchmark.py` to reproduce the experiments used for the paper, as follows:

```bash
python benchmark.py --config benchmark-llama3.yml --results results/ --gpus 0,1
```
where `--gpus` refers to how many gpus to use. If multiple GPUs are specified, each one will be used
to run an individual experiment.
