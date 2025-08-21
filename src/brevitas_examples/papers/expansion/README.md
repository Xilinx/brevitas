# Improving Quantization with Post-Training Model Expansion

📄 [Paper](https://arxiv.org/abs/2503.17513)

```
@article{franco2025improving,
  title={Improving quantization with post-training model expansion},
  author={Franco, Giuseppe and Monteagudo-Lago, Pablo and Colbert, Ian and Fraser, Nicholas and Blott, Michaela},
  journal={arXiv preprint arXiv:2503.17513},
  year={2025}
}
```

Please use `benchmark.py` to reproduce the experiments used for the paper, as follows:

```bash
python benchmark.py --config quarot_star.yaml --results results/ --gpus 0,1
```
where `--gpus` refers to how many gpus to use. If multiple GPUs are specified, each one will be used
to run an individual experiment.
