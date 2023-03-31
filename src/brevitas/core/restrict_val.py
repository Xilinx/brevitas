# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Module

import brevitas
from brevitas.core.function_wrapper import Identity
from brevitas.core.function_wrapper import InplaceLogTwo
from brevitas.core.function_wrapper import LogTwo
from brevitas.core.function_wrapper import PowerOfTwo
from brevitas.core.function_wrapper import RoundSte
from brevitas.core.function_wrapper import ScalarClampMinSte
from brevitas.inject.enum import FloatToIntImplType  # retrocompatibility
from brevitas.inject.enum import RestrictValueType

assert RestrictValueType  # prevent removal of unused import
assert FloatToIntImplType


class _RestrictClampValue(brevitas.jit.ScriptModule):

    def __init__(self, scaling_min_val: Optional[float], restrict_value_impl: Optional[Module]):
        super(_RestrictClampValue, self).__init__()
        if scaling_min_val is not None and scaling_min_val != 0:
            self.clamp_min_ste = ScalarClampMinSte(scaling_min_val)
        else:
            self.clamp_min_ste = Identity()
        if restrict_value_impl is not None:
            self.restrict_value_impl = restrict_value_impl
        else:
            self.restrict_value_impl = Identity()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        x = self.restrict_value_impl(x)
        x = self.clamp_min_ste(x)
        return x


class _RestrictValue(brevitas.jit.ScriptModule):

    def __init__(self, restrict_value_impl: Optional[Module]):
        super(_RestrictValue, self).__init__()
        if restrict_value_impl is not None:
            self.restrict_value_impl = restrict_value_impl
        else:
            self.restrict_value_impl = Identity()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        x = self.restrict_value_impl(x)
        return x


class _ClampValue(brevitas.jit.ScriptModule):

    def __init__(self, scaling_min_val: Optional[float]):
        super(_ClampValue, self).__init__()
        if scaling_min_val is not None and scaling_min_val != 0:
            self.clamp_min_ste = ScalarClampMinSte(scaling_min_val)
        else:
            self.clamp_min_ste = Identity()
        self.min_val = scaling_min_val

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        x = self.clamp_min_ste(x)
        return x


class FloatRestrictValue(brevitas.jit.ScriptModule):

    def __init__(self) -> None:
        super(FloatRestrictValue, self).__init__()

    def restrict_init_float(self, x: float) -> float:
        return x

    def restrict_init_tensor(self, x: Tensor) -> Tensor:
        return x

    def restrict_init_module(self):
        return Identity()

    def restrict_init_inplace_module(self):
        return Identity()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor) -> Tensor:
        return x


class LogFloatRestrictValue(brevitas.jit.ScriptModule):

    def __init__(self):
        super(LogFloatRestrictValue, self).__init__()
        self.power_of_two: Module = PowerOfTwo()

    def restrict_init_float(self, x: float):
        return math.log2(x)

    def restrict_init_tensor(self, x: torch.Tensor):
        return torch.log2(x)

    def restrict_init_module(self):
        return LogTwo()

    def restrict_init_inplace_module(self):
        return InplaceLogTwo()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        x = self.power_of_two(x)
        return x


class IntRestrictValue(brevitas.jit.ScriptModule):

    def __init__(self, restrict_value_float_to_int_impl: Module = RoundSte()):
        super(IntRestrictValue, self).__init__()
        self.float_to_int_impl = restrict_value_float_to_int_impl

    def restrict_init_float(self, x: float):
        return x

    def restrict_init_tensor(self, x: torch.Tensor):
        return x

    def restrict_init_module(self):
        return Identity()

    def restrict_init_inplace_module(self):
        return Identity()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        x = self.float_to_int_impl(x)
        return x


class PowerOfTwoRestrictValue(brevitas.jit.ScriptModule):

    def __init__(self, restrict_value_float_to_int_impl: Module = RoundSte()):
        super(PowerOfTwoRestrictValue, self).__init__()
        self.float_to_int_impl = restrict_value_float_to_int_impl
        self.power_of_two: Module = PowerOfTwo()

    def restrict_init_float(self, x: float):
        return math.log2(x)

    def restrict_init_tensor(self, x: torch.Tensor):
        return torch.log2(x)

    def restrict_init_module(self):
        return LogTwo()

    def restrict_init_inplace_module(self):
        return InplaceLogTwo()

    @brevitas.jit.script_method
    def forward(self, x: torch.Tensor):
        x = self.float_to_int_impl(x)
        x = self.power_of_two(x)
        return x
