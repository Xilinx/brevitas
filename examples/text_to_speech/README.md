# Examples

The models provided in this folder are meant to showcase how to leverage the quantized layers provided by Brevitas,
and by no means a direct mapping to hardware should be assumed.

Below in the table is a list of example pretrained models made available for reference.

| Name         | Cfg                   | Scaling Type               | Last activation bit | Weights | Activations | Pretrained model                                                                                                             | Implementation based on                                              |
|--------------|-----------------------|----------------------------|---------------------|---------|-------------|------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| MelGAN       | quant_melgan_8b       | Floating-point per tensor  | 16 bit              | 8 bit   | 8 bit       | [Generator](https://github.com/Xilinx/brevitas/releases/download/quant_melgan_8b-r0/quant_melgan_8b_generator-8fe7e01f.pth), [Discriminator](https://github.com/Xilinx/brevitas/releases/download/quant_melgan_8b-r0/quant_melgan_8b_discriminator-f1ff0ef6.pth) | [link](https://github.com/seungwonpark/melgan/blob/master/README.md) |


It is highly recommended to setup a virtual environment.

After downloading the LJSpeech1.1 dataset, create the validation folder with the following commands:
```
find /path/to/LJSpeech/LJSpeech-1.1/wavs -type f | head -10 | xargs cp -t /path/to/validation/folder
```

To evaluate a pretrained quantized model on LJSpeech1.1:

 - Install  the MelGAN requirements with `pip install requirements.txt`
 - Make sure you have Brevitas installed
 - Preprocess the dataset with `python preprocess_datasets --model-cfg cfg/quant_melgan_8b --data-path /path/to/validation/folder`
 - Pass the corresponding cfg .ini file as an input to the evaluation script. The required checkpoint will be downloaded automatically. 
 
 For example, for the evaluation on GPU 0:

```
python melgan_val.py --input-folder /path/to/validation/folder --model-cfg cfg/quant_melgan_8b --gpu 0
```
