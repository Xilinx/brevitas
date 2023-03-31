# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from .hadamard_classifier import HadamardClassifier
from .quant_accumulator import ClampQuantAccumulator
from .quant_accumulator import TruncQuantAccumulator
from .quant_activation import QuantHardTanh
from .quant_activation import QuantIdentity
from .quant_activation import QuantReLU
from .quant_activation import QuantSigmoid
from .quant_activation import QuantTanh
from .quant_avg_pool import TruncAdaptiveAvgPool2d
from .quant_avg_pool import TruncAvgPool2d
from .quant_bn import BatchNorm1dToQuantScaleBias
from .quant_bn import BatchNorm2dToQuantScaleBias
from .quant_conv import QuantConv1d
from .quant_conv import QuantConv2d
from .quant_convtranspose import QuantConvTranspose1d
from .quant_convtranspose import QuantConvTranspose2d
from .quant_dropout import QuantDropout
from .quant_eltwise import QuantCat
from .quant_eltwise import QuantEltwiseAdd
from .quant_embedding import QuantEmbedding
from .quant_linear import QuantLinear
from .quant_max_pool import QuantMaxPool1d
from .quant_max_pool import QuantMaxPool2d
from .quant_mha import QuantMultiheadAttention
from .quant_rnn import QuantLSTM
from .quant_rnn import QuantRNN
from .quant_scale_bias import QuantScaleBias
from .quant_scale_bias import ScaleBias
from .quant_upsample import QuantUpsample
from .quant_upsample import QuantUpsamplingBilinear2d
from .quant_upsample import QuantUpsamplingNearest2d
from .target import flexml
