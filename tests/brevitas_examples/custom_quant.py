from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat

quantization_dict = dict()
quantization_dict['weight_quant'] = Int8WeightPerTensorFloat
