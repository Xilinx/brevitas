# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch


def refresh_parameter_quantization_references(model: torch.nn.Module) -> None:
    """Refresh quantizer aliases after FSDP2 replaces model parameters."""
    from brevitas.proxy.parameter_quant import ParameterQuantProxyFromInjector

    for proxy in model.modules():
        if not isinstance(proxy, ParameterQuantProxyFromInjector):
            continue
        if proxy.tensor_quant is None:
            continue
        tracked_parameter_list = proxy.tracked_parameter_list
        for module in proxy.tensor_quant.modules():
            refresh = getattr(module, 'refresh_tracked_parameters', None)
            if refresh is not None:
                refresh(tracked_parameter_list)
