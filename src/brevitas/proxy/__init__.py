# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


from .parameter_quant import WeightQuantProxyFromInjector
from .parameter_quant import DecoupledWeightQuantProxyFromInjector
from .parameter_quant import BiasQuantProxyFromInjector
from .runtime_quant import ActQuantProxyFromInjector
from .runtime_quant import ClampQuantProxyFromInjector
from .runtime_quant import TruncQuantProxyFromInjector
