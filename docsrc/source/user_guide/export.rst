====================
Export Compatibility
====================

Layer / Toolchain
_________________

Quantized linear layers
'''''''''''''''''''''''

.. csv-table::
   :header: "Layer", "PyTorch", "ONNX", "FINN", "XIR", "PyXIR"

   "QuantConv1d", |:heavy_check_mark:|, |:heavy_check_mark:|, |:heavy_check_mark:|, |:heavy_check_mark:|, |:heavy_check_mark:|
   "QuantConv2d", |:heavy_check_mark:|, |:heavy_check_mark:|, |:heavy_check_mark:|, |:heavy_check_mark:|, |:heavy_check_mark:|
   "QuantLinear", |:heavy_check_mark:|, |:heavy_check_mark:|, |:heavy_check_mark:|, |:heavy_check_mark:|, |:heavy_check_mark:|
   "QuantConvTranspose1d", |:x:|, |:x:|, |:x:|, |:heavy_check_mark:|, |:x:|
   "QuantConvTranspose2d", |:x:|, |:x:|, |:x:|, |:heavy_check_mark:|, |:x:|
   "QuantScaleBias", |:x:|, |:x:|, |:x:|, |:x:|, |:x:|
   "BatchNorm2dToQuantScaleBias", |:x:|, |:x:|, |:x:|, |:x:|, |:x:|
   "HadamardClassifier",  |:x:|, |:x:|, |:x:|, |:x:|, |:x:|

Quantized activation layers
'''''''''''''''''''''''''''

.. csv-table::
   :header: "Layer", "PyTorch", "ONNX", "FINN", "XIR", "PyXIR"

   "QuantReLU",
   "QuantIdentity",
   "QuantSigmoid",
   "QuantTanh",
   "QuantHardTanh",


Quantized element-wise layers
'''''''''''''''''''''''''''''

.. csv-table::
   :header: "Layer", "PyTorch", "ONNX", "FINN", "XIR", "PyXIR"

    "QuantEltwiseAdd",
    "QuantCat",


Quantized pooling layers
'''''''''''''''''''''''''''''

.. csv-table::
   :header: "Layer", "PyTorch", "ONNX", "FINN", "XIR", "PyXIR"

    "QuantMaxPool1d",
    "QuantMaxPool2d",
    "QuantAvgPool2d",
    "QuantAdaptiveAvgPool2d",
    "torch.nn.functional.max_pool1d w/ QuantTensor input"
    "torch.nn.functional.max_pool2d w/ QuantTensor input"


Quantized accumulators layers
'''''''''''''''''''''''''''''

.. csv-table::
   :header: "Layer", "PyTorch", "ONNX", "FINN", "XIR", "PyXIR"

   "ClampQuantAccumulator",
   "TruncQuantAccumulator",


Quantized upsampling layers
'''''''''''''''''''''''''''''

.. csv-table::
   :header: "Layer", "PyTorch", "ONNX", "FINN", "XIR", "PyXIR"

   "QuantUpsample",
   "QuantUpsamplingBilinear2d",
   "QuantUpsamplingNearest2d",
