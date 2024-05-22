# Integer-Quantized Super Resolution Experiments with Brevitas

This directory contains scripts demonstrating how to train integer-quantized super resolution models using Brevitas.
Code is also provided to demonstrate accumulator-aware quantization (A2Q) as proposed in our ICCV 2023 paper "[A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance](https://arxiv.org/abs/2308.13504)", as well as A2Q+ as proposed in our paper "[A2Q+: Improving Accumulator-Aware Weight Quantization](https://arxiv.org/abs/2401.10432)", where we introduce and motivate the zero-centered weight quantizer (i.e., `AccumulatorAwareZeroCenterWeightQuant`)

## Experiments

All models are trained on the BSD300 dataset to upsample images by 2x.
Target images are cropped to 512x512.
During training random cropping is applied, along with random vertical and horizontal flips.
During inference center cropping is applied.
Inputs are then downscaled by 2x and then used to train the model directly in the RGB space.
Note that this is a difference from many academic works that train only on the Y-channel in YCbCr format.

| Model Name | Upscale Factor | Weight quantization | Activation quantization | Peak Signal-to-Noise Ratio |
|-----------------------------|----------------|---------------------|-------------------------|----------------------------|
| bicubic_interp | x2 | N/A | N/A | 28.71 |
| [float_espcn_x2](https://github.com/Xilinx/brevitas/releases/download/super_res_r1/float_espcn_x2-2f85a454.pth) | x2 | float32 | float32 | 31.03 |
||
| [quant_espcn_x2_w8a8_base](https://github.com/Xilinx/brevitas/releases/download/super_res_r1/quant_espcn_x2_w8a8_base-f761e4a1.pth) | x2 | int8 | (u)int8 | 30.96 |
| [quant_espcn_x2_w8a8_a2q_32b](https://github.com/Xilinx/brevitas/releases/download/super_res_r1/quant_espcn_x2_w8a8_a2q_32b-85470d9b.pth) | x2 | int8 | (u)int8 | 30.79 |
| [quant_espcn_x2_w8a8_a2q_16b](https://github.com/Xilinx/brevitas/releases/download/super_res_r1/quant_espcn_x2_w8a8_a2q_16b-f9e1da66.pth) | x2 | int8 | (u)int8 | 30.56 |
| [quant_espcn_x2_w8a8_a2q_plus_16b](https://github.com/Xilinx/brevitas/releases/download/super_res_r2/quant_espcn_x2_w8a8_a2q_plus_16b-0ddf46f1.pth) | x2 | int8 | (u)int8 | 31.24 |
||
| [quant_espcn_x2_w4a4_base](https://github.com/Xilinx/brevitas/releases/download/super_res_r1/quant_espcn_x2_w4a4_base-80658e6d.pth) | x2 | int4 | (u)int4 | 30.30 |
| [quant_espcn_x2_w4a4_a2q_32b](https://github.com/Xilinx/brevitas/releases/download/super_res_r1/quant_espcn_x2_w4a4_a2q_32b-8702a412.pth) | x2 | int4 | (u)int4 | 30.27 |
| [quant_espcn_x2_w4a4_a2q_13b](https://github.com/Xilinx/brevitas/releases/download/super_res_r1/quant_espcn_x2_w4a4_a2q_13b-9fff234e.pth) | x2 | int4 | (u)int4 | 30.24 |
| [quant_espcn_x2_w4a4_a2q_plus_13b](https://github.com/Xilinx/brevitas/releases/download/super_res_r2/quant_espcn_x2_w4a4_a2q_plus_13b-6e6d55f0.pth) | x2 | int4 | (u)int4 | 30.95 |


## Train

All models are trained from scratch as follows:
 ```bash
python train_model.py^
    --data_root=./data^
    --model=quant_espcn_x2_w8a8_a2q_32b^
    --batch_size=8^
    --learning_rate=0.001^
    --weight_decay=0.00001^
    --gamma=0.999^
    --step_size=1
 ```

## Evaluate

To evaluate a trained model from a pretrained checkpoint:
```bash
python eval_model.py --data_root=data --use_pretrained --model=quant_espcn_x2_w8a8_a2q_32b
```

To evaluate a trained model from a locally saved checkpoint:
```bash
python eval_model.py --data_root=data --model_path=outputs/model.pth --model=quant_espcn_x2_w8a8_a2q_32b
```
