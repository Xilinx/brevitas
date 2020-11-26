# Examples

The models provided in this folder are meant to showcase how to leverage the quantized layers provided by Brevitas,
and by no means a direct mapping to hardware should be assumed.

Below in the table is a list of example pretrained models made available for reference.

| Name         | Cfg                   | Scaling Type               | Inner layers bit width | Outer layers bit width | WER (Word Error Rate) on dev-other  |  Pretrained model    | Retrained from                |
|--------------|-----------------------|----------------------------|------------------------|------------------------|------------------------|----------------------|-------------------------------|
| Quartznet 8b | quant_quartznet_pertensorscaling_8b  | Floating-point per tensor  | 8 bit | 8 bit | 11.03% | [Encoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_8b-r0/quant_quartznet_encoder_8b-50f12b4b.pth) [Decoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_8b-r0/quant_quartznet_decoder_8b-af09651c.pth) | [link](https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp) |
| Quartznet 8b | quant_quartznet_perchannelscaling_8b | Floating-point per channel | 8 bit | 8 bit | 10.98% | [Encoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_8b-r0/quant_quartznet_encoder_8b-50f12b4b.pth) [Decoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_8b-r0/quant_quartznet_decoder_8b-af09651c.pth) | [link](https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp) |
| Quartznet 4b | quant_quartznet_perchannelscaling_4b | Floating-point per channel | 4 bit | 8 bit | 12.00% | [Encoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_4b-r0/quant_quartznet_encoder_4b-0a46a232.pth) [Decoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_4b-r0/quant_quartznet_decoder_4b-bcbf8c7b.pth) | [link](https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp) |

It is highly recommended to setup a virtual environment.

Download and pre-process the LibriSpeech dataset with the following command:
```
brevitas_quartznet_preprocess --data_root=/path/to/validation/folder --data_set=DEV_OTHER
```

To evaluate a pretrained quantized model on LibriSpeech:

 - Install pytorch from the [Pytorch Website](https://pytorch.org/), and Cython with the following command:
 `python install --upgrade cython`
 - Install SoX (this [guide](https://at.projects.genivi.org/wiki/display/PROJ/Installation+of+SoX+on+different+Platforms)
 may be helpful)
 - After cloning the repository, install Brevitas and QuartzNet requirements with `pip install .[stt]`
 - Pass the name of the model as an input to the evaluation script. The required checkpoint will be downloaded automatically. 
 
 For example, for the evaluation on GPU 0:

```
brevitas_quartznet_val --input-folder /path/to/validation/folder --model quant_quartznet_pertensorscaling_8b --gpu 0 --pretrained
```
