# Integer-Quantized Super Resolution Experiments with Brevitas

This directory contains training scripts to demonstrate how to train integer-quantized super resolution models using [Brevitas](https://github.com/Xilinx/brevitas).
Code is also provided to demonstrate accumulator-aware quantization (A2Q) as proposed in our paper "[Quantized Neural Networks for Low-Precision Accumulation with Guaranteed Overflow Avoidance]"(https://arxiv.org/abs/2301.13376).

## Experiments

| Name                        | Upscale Factor | Weight quantization | Activation quantization | Peak Signal-to-Noise Ratio |
|-----------------------------|----------------|---------------------|-------------------------|----------------------------|
| float_espcn_x2              | x2             | float32             | float32                 | 30.37                      |
| quant_espcn_x2_w8a8_base    | x2             | int8                | (u)int8                 | 30.16                      |
| quant_espcn_x2_w8a8_a2q_32b | x2             | int8                | (u)int8                 | 30.80                      |
| quant_espcn_x2_w8a8_a2q_16b | x2             | int8                | (u)int8                 | 29.25                      |
| bicubic_interp              | x2             | N/A                 | N/A                     | 28.71                      |


## Train

To start training a model from scratch (*e.g.*, `quant_espcn_x2_w8a8_a2q_32b`) run:
 ```bash
python train_model.py --data_root=data --model=quant_espcn_x2_w8a8_a2q_32b
 ```

## Evaluate

To evaluate a trained model from a saved checkpoint:
```bash
python eval_model.py --data_root=data --model_path=outputs/model.pth --model=quant_espcn_x2_w8a8_a2q_32b
```
