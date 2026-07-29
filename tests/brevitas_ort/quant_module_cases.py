# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Model builders for the ORT integration tests.

These functions construct the ``nn.Module`` under test from a sampled configuration
object (see :mod:`tests.brevitas_ort.sampling`). They used to be ``pytest_cases`` case
classes producing the full parameter cross-product; that enumeration has been replaced
by seeded random sampling, so all that remains here is the model-construction logic.
"""

from torch import nn

from brevitas.nn.quant_activation import QuantIdentity
from brevitas.nn.quant_avg_pool import TruncAvgPool2d
from brevitas.nn.quant_rnn import QuantLSTM
from brevitas.quant.scaled_int import Int32Bias

from .common import *
from .sampling import AvgPoolConfig
from .sampling import LSTMConfig
from .sampling import WBIOLConfig


def build_wbiol_model(cfg: WBIOLConfig) -> nn.Module:
    # FP8 export and FLOOR rounding need non-deterministic algorithms; everything else
    # runs deterministically. Validity of the bit-widths is enforced during sampling.
    if cfg.is_fp8 or cfg.rounding_type == 'floor':
        torch.use_deterministic_algorithms(False)
    else:
        torch.use_deterministic_algorithms(True)

    impl = cfg.impl
    if impl is QuantLinear:
        layer_kwargs = {'in_features': IN_CH, 'out_features': OUT_CH}
    else:
        layer_kwargs = {'in_channels': IN_CH, 'out_channels': OUT_CH, 'kernel_size': KERNEL_SIZE}

    bias_quantizer = None if (cfg.is_fp8 or cfg.is_dynamic) else Int32Bias
    # Required because of numpy error with FP8 data type. Export itself works fine.
    return_quant_tensor = False if cfg.is_fp8 else True

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv = impl(
                **layer_kwargs,
                bias=True,
                weight_quant=cfg.weight_quant,
                input_quant=cfg.io_quant,
                output_quant=cfg.io_quant,
                weight_bit_width=cfg.weight_bit_width,
                input_bit_width=cfg.input_bit_width,
                output_bit_width=cfg.output_bit_width,
                bias_quant=bias_quantizer,
                weight_float_to_int_impl_type=cfg.rounding_type,
                return_quant_tensor=return_quant_tensor)
            self.conv.weight.data.uniform_(-0.01, 0.01)

        def forward(self, x):
            return self.conv(x)

    torch.random.manual_seed(SEED)
    return Model()


def build_avgpool_model(cfg: AvgPoolConfig) -> nn.Module:

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.in_quant = QuantIdentity(signed=cfg.input_signed, return_quant_tensor=True)
            self.quant_avg_pool = TruncAvgPool2d(
                kernel_size=3,
                stride=2,
                bit_width=cfg.output_bit_width,
                float_to_int_impl_type='round')

        def forward(self, x):
            return self.quant_avg_pool(self.in_quant(x))

    torch.random.manual_seed(SEED)
    return Model()


def build_lstm_model(cfg: LSTMConfig) -> nn.Module:
    if cfg.bidirectional == 'shared_input_hidden':
        bidirectional = True
        shared_input_hidden = True
    else:
        bidirectional = cfg.bidirectional
        shared_input_hidden = False

    quant_kwargs = {}
    if cfg.is_quant:
        quant_kwargs['weight_quant'] = cfg.weight_quant
        quant_kwargs['weight_bit_width'] = cfg.weight_bit_width
    else:
        quant_kwargs['weight_quant'] = None

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.lstm = QuantLSTM(
                input_size=IN_CH,
                hidden_size=OUT_CH,
                **quant_kwargs,
                bias_quant=None,
                io_quant=None,
                gate_acc_quant=None,
                sigmoid_quant=None,
                tanh_quant=None,
                cell_state_quant=None,
                batch_first=False,  # ort doesn't support batch_first=True (layout = 1)
                num_layers=cfg.num_layers,
                bidirectional=bidirectional,
                shared_input_hidden_weights=shared_input_hidden,
                coupled_input_forget_gates=cfg.cifg)

        def forward(self, x):
            return self.lstm(x)

    torch.random.manual_seed(SEED)
    return Model()
