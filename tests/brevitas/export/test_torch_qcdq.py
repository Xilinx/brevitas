# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import pytest
import torch

from brevitas.export import export_torch_qcdq
from tests.marker import jit_disabled_for_export
from tests.marker import requires_pt_ge

from .quant_module_fixture import *


@requires_pt_ge('1.9.1')
@jit_disabled_for_export()
def test_pytorch_qcdq_export(
        quant_module,
        quant_module_impl,
        weight_act_quantizers,
        input_bit_width,
        weight_bit_width,
        output_bit_width,
        bias_bit_width,
        bias_quantizer):

    weight_act_quantizers_name, _ = weight_act_quantizers
    bias_quantizer_name, _ = bias_quantizer

    if 'asymmetric_act' in weight_act_quantizers_name and (input_bit_width > 8 or output_bit_width > 8):
        pytest.skip("Unsigned zero point supported on 8b or less.")
    if 'asymmetric_weight' in weight_act_quantizers_name and weight_bit_width > 8:
        pytest.skip("Unsigned zero point supported on 8b or less.")
    if 'internal_scale' in bias_quantizer_name and bias_bit_width == 32:
        pytest.skip("This combination is prone to numerical errors as the scale gets too small.")

    if quant_module_impl == QuantLinear:
        in_size = (1, IN_CH)
    elif quant_module_impl == QuantConv1d or quant_module_impl == QuantConvTranspose1d:
        in_size = (1, IN_CH, FEATURES)
    else:
        in_size = (1, IN_CH, FEATURES, FEATURES)

    inp = torch.randn(in_size)
    quant_module(inp)  # Collect scale factors
    quant_module.eval()
    inp = torch.randn(in_size) * IN_SCALE + IN_MEAN  # redefine inp for testing
    out = quant_module(inp)
    pytorch_qcdq_model = export_torch_qcdq(quant_module, args=inp)
    torchscript_out = pytorch_qcdq_model(inp)
    torchscript_out_value = torchscript_out[0]
    tolerance = TOLERANCE * out.scale
    del pytorch_qcdq_model
    assert torch.allclose(out, torchscript_out_value, atol=tolerance)
