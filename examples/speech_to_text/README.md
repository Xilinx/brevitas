# Examples

The models provided in this folder are meant to showcase how to leverage the quantized layers provided by Brevitas,
and by no means a direct mapping to hardware should be assumed.

Below in the table is a list of example pretrained models made available for reference.

| Name         | Cfg                   | Scaling Type               | Inner layers bit width | Outer layers bit width | WER (Word Error Rate) on dev-other  |  Pretrained model    | Retrained from                |
|--------------|-----------------------|----------------------------|------------------------|------------------------|------------------------|----------------------|-------------------------------|
| Quartznet 8b | quant_quartznet_pertensorscaling_8b  | Floating-point per tensor  | 8 bit | 8 bit | 11.03% | [Encoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_8b-r0/quant_quartznet_encoder_8b-50f12b4b.pth) [Decoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_8b-r0/quant_quartznet_decoder_8b-af09651c.pth) | [link](https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp) |
| Quartznet 8b | quant_quartznet_perchannelscaling_8b | Floating-point per channel | 8 bit | 8 bit | 10.98% | [Encoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_8b-r0/quant_quartznet_encoder_8b-50f12b4b.pth) [Decoder](https://github.com/Xilinx/brevitas/releases/download/quant_quartznet_8b-r0/quant_quartznet_decoder_8b-af09651c.pth) | [link](https://ngc.nvidia.com/catalog/models/nvidia:quartznet_15x5_ls_sp) |

It is highly recommended to setup a virtual environment.

Download and pre-process the LibriSpeech dataset with the following command:
```
python utilities/get_librispeech_data.py --data_root=/path/to/validation/folder --data_set=DEV_OTHER
```

To evaluate a pretrained quantized model on LibriSpeech:

 - Install pytorch from the [Pytorch Website](https://pytorch.org/), and Cython with the following command:
 `python install --upgrade cython`
 - Install  the Quartznet requirements with `pip install requirements.txt`
 - Make sure you have Brevitas installed
 - Pass the corresponding cfg .ini file as an input to the evaluation script. The required checkpoint will be downloaded automatically. 
 
 For example, for the evaluation on GPU 0:

```
python quartznet_val.py --input-folder /path/to/validation/folder --model-cfg cfg/quant_quartznet_pertensorscaling_8b.ini --gpu 0
```
